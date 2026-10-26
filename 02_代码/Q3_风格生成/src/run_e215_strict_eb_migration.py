"""E215 all-data post-selection EB/Mahalanobis geometry checkpoint.

This stage does not search frozen sensory proxies or certify chemical safety.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

from eb_migration import EBStyleGeometry, migration_path
from q1_interface import FrozenQ1V5OOF
from q3_constraints import audit_concentrations
from constrained_migration import stored_from_concentrations
from run_e204_constrained_migration import TargetGeometry, _direct_vectors


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = Path("/home/mary/mathorcup")
RESULT = ROOT / "03_结果/Q3_风格生成"
CONFIG = Path(__file__).resolve().parents[1] / "configs/E215_strict_eb_migration.yaml"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    required = {
        "experiment": "E215", "scope": "full_data_postselection_design_not_outer_validation",
        "source_style": "Pils/Lager", "target_styles": ["Tripel", "Kriek", "Lambic"],
        "scenario": "C1", "mean_kappa": 15.0, "covariance_kappa": 30.0,
        "covariance_estimator": "LedoitWolf_shrunk_to_global",
        "path_lambdas": [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0],
        "minimum_relative_target_gain": 0.01, "source_count": 11,
        "target_pair_count": 33, "direct_defect_evidence": "E200_registered_known_only",
        "q1_retrained": False, "nnls_refit": False, "gpu_used": False,
        "kaggle_used_for_training": False, "sensory_proxy_used_in_geometry_stage": False,
        "chemical_safety_certified": False,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"E215 preregistered setting changed: {key}")
    return True


def validate_geometry_outputs(endpoints, paths, config):
    if len(endpoints) != 33 or len(paths) != 264:
        raise ValueError("E215 requires 11 sources x 3 targets x 8 path positions")
    if endpoints.duplicated(["beer", "target_style"]).any() or paths.duplicated(["beer", "target_style", "lambda"]).any():
        raise ValueError("duplicate source-target or path key")
    if endpoints.beer.nunique() != 11 or set(endpoints.target_style) != set(config["target_styles"]):
        raise ValueError("E215 target or source set incomplete")
    expected = np.array([0.0] + config["path_lambdas"], dtype=float)
    for (_, _), group in paths.groupby(["beer", "target_style"]):
        group = group.sort_values("lambda")
        if len(group) != 8 or not np.array_equal(group["lambda"].to_numpy(float), expected):
            raise ValueError("E215 fixed path values changed")
        if not np.array_equal(group.source_identity.to_numpy(bool), expected == 0.0):
            raise ValueError("lambda=0 must be the only source identity row")
    if set(map(tuple, endpoints[["beer", "target_style"]].to_numpy())) != set(map(tuple, paths[["beer", "target_style"]].drop_duplicates().to_numpy())):
        raise ValueError("endpoint and path pair sets differ")
    return True


def _candidate_concentrations(stored, scales):
    stored = np.asarray(stored, float)
    scales = np.asarray(scales, str)
    values = stored.copy()
    values[scales == "log10"] = np.power(10.0, stored[scales == "log10"])
    return values


def _path_status(n_failed, n_unknown, target_gain):
    if n_failed:
        return "rejected_direct_constraint"
    if not target_gain:
        return "no_target_gain"
    if n_unknown:
        return "needs_direct_verification"
    return "needs_sensory_and_support_review"


def run():
    started = time.perf_counter()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    RESULT.mkdir(parents=True, exist_ok=True)
    q1 = FrozenQ1V5OOF(CANONICAL)
    chemistry_path = ROOT / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"
    source_path = RESULT / "q3_source_defects_E200.csv"
    defect_path = RESULT / "q3_defect_registry_E200.csv"
    contract_path = RESULT / "q3_field_contract_E200.csv"
    e200_manifest = RESULT / "q3_run_manifest_E200.json"
    e204_manifest = RESULT / "q3_run_manifest_E204.json"
    e204_summary_path = RESULT / "q3_migration_summary_E204.csv"
    inputs = [CONFIG, Path(__file__), Path(__file__).with_name("eb_migration.py"),
              Path(__file__).with_name("constrained_migration.py"),
              Path(__file__).with_name("q3_constraints.py"), chemistry_path, source_path,
              defect_path, contract_path, e200_manifest, e204_manifest, e204_summary_path]
    input_hashes = {str(path.resolve()): sha(path) for path in inputs}
    manifest_path = RESULT / "q3_run_manifest_E215_geometry.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") != "complete" or previous.get("input_sha256") != input_hashes or previous.get("q1_final_oof_sha256") != q1.signature:
            raise ValueError("E215 geometry checkpoint input identity changed")
        for name, digest in previous["output_sha256"].items():
            if sha(RESULT / name) != digest:
                raise ValueError("E215 geometry checkpoint output changed")
        print("E215 geometry complete; frozen outputs verified without rewriting", flush=True)
        return previous
    if json.loads(e200_manifest.read_text())["status"] != "complete" or json.loads(e204_manifest.read_text())["status"] != "complete":
        raise ValueError("audited E200/E204 inputs required")
    chemistry = pd.read_csv(chemistry_path)
    features = [c for c in chemistry.columns if c not in ["beer", "beer_id", "tasting_category_fine"]]
    if len(chemistry) != 250 or len(features) != 231 or not chemistry.beer.is_unique:
        raise ValueError("official Q3 chemistry contract changed")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x = imputer.fit_transform(chemistry[features].to_numpy(float))
    styles = chemistry.tasting_category_fine.to_numpy(str)
    model = EBStyleGeometry().fit(x, styles, mean_kappa=config["mean_kappa"], covariance_kappa=config["covariance_kappa"])
    global_mean = x.mean(axis=0)
    global_scale = np.std(x, axis=0)
    global_scale = np.where(global_scale > 1e-12, global_scale, 1.0)
    source_table = pd.read_csv(source_path)
    defect_table = pd.read_csv(defect_path)
    field_contract = pd.read_csv(contract_path)
    source_positions = np.flatnonzero(styles == config["source_style"])
    if len(source_positions) != config["source_count"]:
        raise ValueError("registered Pils/Lager source count changed")
    geometries = {style: TargetGeometry(x[styles == style], global_mean, global_scale) for style in config["target_styles"]}
    endpoints = []
    paths = []
    spectra = []
    for pos in source_positions:
        beer = int(chemistry.beer.iloc[pos])
        source = x[pos]
        _, direct_fields, _, direct_scales, source_values, source_known, thresholds = _direct_vectors(
            beer, source_table, defect_table, field_contract
        )
        direct_indices = np.array([features.index(name) for name in direct_fields], dtype=int)
        known_indices = direct_indices[source_known]
        upper_stored = stored_from_concentrations(source_values[source_known], np.asarray(direct_scales)[source_known])
        for style in config["target_styles"]:
            anchor = model.anchor(source, config["source_style"], style)
            endpoint, dual = model.project(anchor, style, known_indices, upper_stored)
            trajectory = migration_path(source, endpoint, config["path_lambdas"], include_identity=True)
            geometry = geometries[style]
            source_distance = float(geometry.distance(source)[0])
            target_block = x[styles == style]
            lower, upper = target_block.min(axis=0), target_block.max(axis=0)
            for path_index, (lam, candidate) in enumerate(zip([0.0] + config["path_lambdas"], trajectory)):
                concentrations = _candidate_concentrations(candidate[direct_indices], direct_scales)
                audit = audit_concentrations(source_values, concentrations, source_known, "C1", threshold=thresholds)
                n_failed = int((audit.passed == False).fillna(False).sum())
                n_unknown = int(audit.passed.isna().sum())
                distance = float(geometry.distance(candidate)[0])
                gain = float((source_distance - distance) / max(source_distance, 1e-12))
                outside = (candidate < lower) | (candidate > upper)
                row = {"beer": beer, "source_style": config["source_style"], "target_style": style,
                    "path_index": path_index, "lambda": float(lam), "source_identity": bool(lam == 0.0),
                    "source_target_distance": source_distance, "candidate_target_distance": distance,
                    "relative_target_gain": gain, "target_gain": bool(gain >= config["minimum_relative_target_gain"]),
                    "source_change_rms": float(np.sqrt(np.mean(((candidate - source) / global_scale) ** 2))),
                    "target_support_outside_cell_fraction": float(outside.mean()),
                    "n_direct_known": int(source_known.sum()), "n_direct_unknown": n_unknown,
                    "n_direct_failed": n_failed, "status": _path_status(n_failed, n_unknown, gain >= config["minimum_relative_target_gain"]),
                    "physical_safety_certified": False,
                    "scope": config["scope"]}
                paths.append(row)
                spectra.append({"beer": beer, "target_style": style, "path_index": path_index,
                                "lambda": float(lam), **dict(zip(features, candidate))})
                if lam == 1.0:
                    endpoints.append({**row, "unconstrained_anchor_target_distance": float(geometry.distance(anchor)[0]),
                        "projection_change_rms": float(np.sqrt(np.mean(((endpoint - anchor) / global_scale) ** 2))),
                        "n_active_direct_bounds": int((dual > 1e-9).sum())})
            if not np.array_equal(trajectory[0], source):
                raise ValueError("lambda=0 source spectrum changed")
    endpoint_table = pd.DataFrame(endpoints)
    path_table = pd.DataFrame(paths)
    spectrum_table = pd.DataFrame(spectra)
    validate_geometry_outputs(endpoint_table, path_table, config)
    old = pd.read_csv(e204_summary_path)
    old = old[(old.scenario == "C1") & (old.strategy == "coordinate_projection") & (old["lambda"] == 1.0)]
    comparison = endpoint_table.merge(old[["beer", "target_style", "relative_target_gain", "source_change_rms", "n_failed", "n_unknown"]],
        on=["beer", "target_style"], how="left", validate="one_to_one", suffixes=("_new", "_e204"))
    if len(comparison) != 33 or comparison.relative_target_gain_e204.isna().any():
        raise ValueError("E204 descriptive reference coverage incomplete")
    comparison["relative_target_gain_delta"] = comparison.relative_target_gain_new - comparison.relative_target_gain_e204
    outputs = []
    for name, frame in [("q3_geometry_endpoints_E215.csv", endpoint_table),
                        ("q3_geometry_paths_E215.csv", path_table),
                        ("q3_geometry_spectra_E215.csv.gz", spectrum_table),
                        ("q3_geometry_vs_E204_E215.csv", comparison)]:
        path = RESULT / name
        frame.to_csv(path, index=False, compression="gzip" if name.endswith(".gz") else None)
        outputs.append(path)
    for path, digest in input_hashes.items():
        if sha(path) != digest:
            raise ValueError("E215 frozen input changed during geometry computation")
    manifest = {"experiment_id": "E215", "status": "complete", "scope": config["scope"],
        "n_source_target_pairs": 33, "n_nonzero_paths": 231, "n_paths_including_source": 264,
        "q1_final_oof_sha256": q1.signature, "q1_retrained": False, "nnls_refit": False,
        "sensory_proxy_used": False, "gpu_used": False, "kaggle_used_for_training": False,
        "physical_safety_certified": False, "input_sha256": input_hashes,
        "output_sha256": {path.name: sha(path) for path in outputs},
        "elapsed_seconds": time.perf_counter() - started}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("sha256")}, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    run()
