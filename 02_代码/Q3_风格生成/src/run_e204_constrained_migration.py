"""E204 full-data post-selection constrained migration and rejection audit."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer

from constrained_migration import (
    concentrations_from_stored,
    direct_limits,
    empirical_quantile_anchor,
    pareto_mask,
    project_direct_constraints,
)
from gaussian_baselines import StyleGaussianGenerator
from q1_interface import FrozenQ1V5OOF
from q3_constraints import audit_concentrations, summarize_migration


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = Path("/home/mary/mathorcup")
RESULT = ROOT / "03_结果/Q3_风格生成"
CONFIG = Path(__file__).resolve().parents[1] / "configs/E204_constrained_migration.yaml"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_formal_config(config):
    if config.get("source_style") != "Pils/Lager":
        raise ValueError("registered source style changed")
    if config.get("target_styles") != ["Tripel", "Kriek", "Lambic"]:
        raise ValueError("registered target styles changed")
    if config.get("distribution_model") != "style_copula_shrunk":
        raise ValueError("audited Copula primary changed")
    if config.get("scenarios") != ["C1", "C2", "C3"]:
        raise ValueError("registered constraint scenarios changed")
    lambdas = np.asarray(config.get("path_lambdas", []), float)
    if lambdas.size < 2 or not np.isfinite(lambdas).all() or (lambdas <= 0).any() or (lambdas > 1).any():
        raise ValueError("nonzero path lambdas in (0,1] required")
    if int(config.get("generated_pool_per_style", 0)) < 500:
        raise ValueError("target structural pool too small")
    gain = float(config.get("minimum_relative_target_gain", np.nan))
    if not np.isfinite(gain) or not 0 < gain < 1:
        raise ValueError("positive relative target gain required")
    if config.get("stale_hops_proxy") != "deferred_to_E205":
        raise ValueError("E204 cannot self-certify stale-hops risk")
    if config.get("gpu_used") is not False:
        raise ValueError("formal E204 is a CPU statistical design")


def verify_e204_gate(result_dir, *, expected_q1_oof_sha256):
    result_dir = Path(result_dir)
    manifest_path = result_dir / "q3_run_manifest_E203.json"
    audit_path = result_dir / "q3_e203_audit_manifest.json"
    decision_path = result_dir / "q3_e203_replacement_decision.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_id") != "E203"
        or manifest.get("status") != "complete"
        or manifest.get("n_outer_folds") != 30
        or audit.get("status") != "complete"
    ):
        raise ValueError("completed and audited E203 required")
    if manifest.get("signature", {}).get("q1_final_oof_sha256") != expected_q1_oof_sha256:
        raise ValueError("Q1 final OOF identity changed after E203")
    if (
        decision.get("replace_copula_baseline") is not False
        or decision.get("e204_primary_distribution_model") != "style_copula_shrunk"
        or decision.get("e204_gate") != "open_with_copula_primary_and_lowrank_ablation"
    ):
        raise ValueError("E204 Copula gate is not open")
    return {
        "distribution_model": "style_copula_shrunk",
        "e203_manifest_sha256": sha(manifest_path),
        "e203_audit_sha256": sha(audit_path),
        "e203_decision_sha256": sha(decision_path),
    }


class TargetGeometry:
    def __init__(self, target, global_mean, global_scale):
        standardized = (np.asarray(target, float) - global_mean) / global_scale
        self.mean = standardized.mean(axis=0)
        self.precision = LedoitWolf().fit(standardized).precision_
        self.reference = standardized
        self.global_mean = global_mean
        self.global_scale = global_scale
        self.n_features = standardized.shape[1]

    def distance(self, values):
        standardized = (np.asarray(values, float) - self.global_mean) / self.global_scale
        delta = np.atleast_2d(standardized) - self.mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, self.precision, delta), 0.0) / self.n_features)

    def nearest_rms(self, values):
        standardized = (np.atleast_2d(np.asarray(values, float)) - self.global_mean) / self.global_scale
        return cdist(standardized, self.reference).min(axis=1) / np.sqrt(self.n_features)


def _direct_vectors(beer, source_table, defect_table, feature_contract):
    direct = defect_table[defect_table.evidence_type == "direct_chemical"].copy()
    source = source_table[source_table.beer == beer].set_index("feature_code")
    contract = feature_contract.set_index("feature_code")
    features = direct.feature_code.tolist()
    defects = direct.defect.tolist()
    scales = [contract.loc[feature, "scale_in_1_csv"] for feature in features]
    source_values = np.array([source.loc[feature, "source_concentration"] for feature in features], float)
    source_known = np.array([bool(source.loc[feature, "source_known"]) for feature in features], bool)
    thresholds = direct.threshold_median.to_numpy(float)
    return direct, features, defects, scales, source_values, source_known, thresholds


def _candidate_direct(candidate, feature_indices, scales):
    return concentrations_from_stored(np.asarray(candidate)[feature_indices], scales)


def _structural_candidate(pool, anchor, source, scale, feature_indices, scales, source_values, source_known, thresholds, scenario):
    direct = np.vstack([_candidate_direct(row, feature_indices, scales) for row in pool])
    limits = direct_limits(source_values, source_known, thresholds, scenario)
    known = np.isfinite(limits)
    feasible = np.ones(len(pool), dtype=bool)
    if known.any():
        tolerance = 1e-9 + 1e-7 * np.abs(source_values[known])
        feasible = np.all(direct[:, known] <= limits[known] + tolerance, axis=1)
    if not feasible.any():
        return None
    candidates = pool[feasible]
    anchor_distance = np.sqrt(np.mean(((candidates - anchor) / scale) ** 2, axis=1))
    source_change = np.sqrt(np.mean(((candidates - source) / scale) ** 2, axis=1))
    return candidates[np.argmin(anchor_distance + 0.1 * source_change)]


def run():
    started = time.perf_counter()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_formal_config(config)
    q1 = FrozenQ1V5OOF(CANONICAL)
    entry_gate = verify_e204_gate(RESULT, expected_q1_oof_sha256=q1.signature)
    attachment = ROOT / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"
    chemistry = pd.read_csv(attachment)
    features = [column for column in chemistry if column not in ["beer", "beer_id", "tasting_category_fine"]]
    if len(chemistry) != 250 or len(features) != 231:
        raise ValueError("Q3 input contract changed")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x = imputer.fit_transform(chemistry[features].to_numpy(float))
    styles = chemistry.tasting_category_fine.to_numpy(str)
    global_mean = x.mean(axis=0)
    global_scale = np.std(x, axis=0)
    global_scale = np.where(global_scale > 1e-12, global_scale, 1.0)
    source_table = pd.read_csv(RESULT / "q3_source_defects_E200.csv")
    defect_table = pd.read_csv(RESULT / "q3_defect_registry_E200.csv")
    feature_contract = pd.read_csv(RESULT / "q3_field_contract_E200.csv")
    model = StyleGaussianGenerator().fit(x, styles, mean_shrinkage=0.25)
    pool_frames = []
    pools = {}
    geometries = {}
    for style_index, style in enumerate(config["target_styles"]):
        pool = model.generate(
            style,
            int(config["generated_pool_per_style"]),
            method=config["distribution_model"],
            seed=int(config["seed"]) + style_index,
        )["values"]
        pools[style] = pool
        geometries[style] = TargetGeometry(x[styles == style], global_mean, global_scale)
        frame = pd.DataFrame(pool, columns=features)
        frame.insert(0, "sample_id", np.arange(len(frame)))
        frame.insert(0, "style", style)
        frame.insert(0, "scope", "full_data_postselection_target_pool")
        pool_frames.append(frame)
    pool_path = RESULT / "q3_target_pool_E204.csv.gz"
    pd.concat(pool_frames, ignore_index=True).to_csv(pool_path, index=False, compression="gzip")

    source_positions = np.flatnonzero(styles == config["source_style"])
    if len(source_positions) != 11:
        raise ValueError("registered Pils/Lager source count changed")
    summaries = []
    spectra = []
    direct_audits = []
    minimum_gain = float(config["minimum_relative_target_gain"])
    for source_position in source_positions:
        beer = int(chemistry.beer.iloc[source_position])
        source = x[source_position]
        direct, direct_features, defects, direct_scales, source_values, source_known, thresholds = _direct_vectors(
            beer, source_table, defect_table, feature_contract
        )
        feature_indices = [features.index(feature) for feature in direct_features]
        source_reference = x[source_positions[source_positions != source_position]]
        for target_style in config["target_styles"]:
            target_reference = x[styles == target_style]
            anchor = empirical_quantile_anchor(source, source_reference, target_reference)
            geometry = geometries[target_style]
            source_distance = float(geometry.distance(source)[0])
            for scenario in config["scenarios"]:
                projected_direct, _ = project_direct_constraints(
                    anchor[feature_indices], source_values, source_known, thresholds, direct_scales, scenario
                )
                coordinate = anchor.copy()
                coordinate[feature_indices] = projected_direct
                structural = _structural_candidate(
                    pools[target_style],
                    anchor,
                    source,
                    global_scale,
                    feature_indices,
                    direct_scales,
                    source_values,
                    source_known,
                    thresholds,
                    scenario,
                )
                repair_direct, _ = project_direct_constraints(
                    source[feature_indices], source_values, source_known, thresholds, direct_scales, scenario
                )
                repair = source.copy()
                repair[feature_indices] = repair_direct
                candidates = [
                    ("source_unchanged", 0.0, source),
                    ("defect_only_repair", 0.0, repair),
                    ("unconstrained_rank_transport", 1.0, anchor),
                    ("coordinate_projection", 1.0, coordinate),
                ]
                base = structural if structural is not None else coordinate
                if structural is not None:
                    candidates.append(("structural_nearest", 1.0, structural))
                else:
                    summaries.append({
                        "beer": beer,
                        "source_style": config["source_style"],
                        "target_style": target_style,
                        "scenario": scenario,
                        "strategy": "structural_nearest",
                        "lambda": 1.0,
                        "candidate_available": False,
                        "status": "rejected_no_structural_candidate",
                        "physical_safety_certified": False,
                    })
                for value in config["path_lambdas"]:
                    candidate = (1.0 - float(value)) * source + float(value) * base
                    projected, _ = project_direct_constraints(
                        candidate[feature_indices], source_values, source_known, thresholds, direct_scales, scenario
                    )
                    candidate[feature_indices] = projected
                    candidates.append(("constrained_path", float(value), candidate))

                for strategy, path_lambda, candidate in candidates:
                    candidate_concentration = _candidate_direct(candidate, feature_indices, direct_scales)
                    audit = audit_concentrations(
                        source_values,
                        candidate_concentration,
                        source_known,
                        scenario,
                        threshold=thresholds,
                    )
                    candidate_distance = float(geometry.distance(candidate)[0])
                    relative_gain = (source_distance - candidate_distance) / max(source_distance, 1e-12)
                    target_gain = bool(relative_gain >= minimum_gain)
                    source_change = float(np.sqrt(np.mean(((candidate - source) / global_scale) ** 2)))
                    summary = summarize_migration(audit, target_gain=target_gain, proxy_nonworsening=None)
                    row = {
                        "beer": beer,
                        "source_style": config["source_style"],
                        "target_style": target_style,
                        "scenario": scenario,
                        "strategy": strategy,
                        "lambda": path_lambda,
                        "candidate_available": True,
                        "source_target_distance": source_distance,
                        "candidate_target_distance": candidate_distance,
                        "relative_target_gain": relative_gain,
                        "target_gain": target_gain,
                        "source_change_rms": source_change,
                        "nearest_target_rms": float(geometry.nearest_rms(candidate)[0]),
                        "proxy_nonworsening": pd.NA,
                        **summary,
                    }
                    summaries.append(row)
                    spectrum = {"beer": beer, "target_style": target_style, "scenario": scenario, "strategy": strategy, "lambda": path_lambda}
                    spectrum.update(dict(zip(features, candidate)))
                    spectra.append(spectrum)
                    for defect, feature, audit_row in zip(defects, direct_features, audit.to_dict("records")):
                        direct_audits.append({
                            "beer": beer,
                            "target_style": target_style,
                            "strategy": strategy,
                            "lambda": path_lambda,
                            "defect": defect,
                            "feature_code": feature,
                            "source_known": bool(source_known[defects.index(defect)]),
                            **audit_row,
                        })

    summary_table = pd.DataFrame(summaries)
    summary_table["pareto_front"] = False
    available = summary_table.candidate_available.eq(True)
    eligible_for_front = (
        available
        & summary_table["target_gain"].eq(True)
        & pd.to_numeric(summary_table["n_failed"], errors="coerce").fillna(1).eq(0)
    )
    for _, index in summary_table[eligible_for_front].groupby(["beer", "target_style", "scenario"]).groups.items():
        subset = summary_table.loc[index]
        summary_table.loc[index, "pareto_front"] = pareto_mask(
            subset.candidate_target_distance.to_numpy(float), subset.source_change_rms.to_numpy(float)
        )
    summary_path = RESULT / "q3_migration_summary_E204.csv"
    spectra_path = RESULT / "q3_migration_spectra_E204.csv.gz"
    audit_path = RESULT / "q3_migration_direct_audit_E204.csv"
    summary_table.to_csv(summary_path, index=False)
    pd.DataFrame(spectra).to_csv(spectra_path, index=False, compression="gzip")
    pd.DataFrame(direct_audits).to_csv(audit_path, index=False)
    aggregate = (
        summary_table[available]
        .groupby(["scenario", "strategy"], as_index=False)
        .agg(
            n_candidates=("beer", "size"),
            target_gain_rate=("target_gain", "mean"),
            mean_relative_target_gain=("relative_target_gain", "mean"),
            mean_source_change_rms=("source_change_rms", "mean"),
            mean_target_distance=("candidate_target_distance", "mean"),
            total_direct_failures=("n_failed", "sum"),
            total_direct_unknown=("n_unknown", "sum"),
            pareto_count=("pareto_front", "sum"),
        )
    )
    aggregate_path = RESULT / "q3_migration_aggregate_E204.csv"
    aggregate.to_csv(aggregate_path, index=False)
    decision = {
        "status": "direct_constraint_migration_complete_proxy_pending",
        "distribution_model": config["distribution_model"],
        "n_sources": len(source_positions),
        "n_targets": len(config["target_styles"]),
        "scenarios": config["scenarios"],
        "stale_hops_proxy_evaluated": False,
        "sensory_profiles_generated": False,
        "physical_safety_certified": False,
        "q1_retrained": False,
        "gpu_used": False,
        "notes": [
            "C1/C2/C3分别审计，未知源浓度和未知阈值不计为通过",
            "零迁移和仅修复缺陷均不能凭直接约束通过就算迁移成功",
            "E205接入独立感官画像与Stale hops代理前不产生最终可用候选",
        ],
    }
    decision_path = RESULT / "q3_migration_decision_E204.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs = [pool_path, summary_path, spectra_path, audit_path, aggregate_path, decision_path]
    manifest = {
        "experiment_id": "E204",
        "status": "complete",
        "scope": config["scope"],
        "signature": {
            "input_sha256": sha(attachment),
            "config_sha256": sha(CONFIG),
            "source_sha256": {
                path.name: sha(path)
                for path in [Path(__file__), Path(__file__).with_name("constrained_migration.py"), Path(__file__).with_name("q3_constraints.py"), Path(__file__).with_name("gaussian_baselines.py")]
            },
            "q1_final_model": q1.registry["final_model"],
            "q1_final_oof_sha256": q1.signature,
            "entry_gate": entry_gate,
        },
        "python": platform.python_version(),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_used": False,
        "decision": decision,
        "output_sha256": {path.name: sha(path) for path in outputs},
    }
    manifest_path = RESULT / "q3_run_manifest_E204.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    run()
