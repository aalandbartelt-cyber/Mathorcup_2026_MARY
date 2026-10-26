"""E216 two-sided direct-defect box migration, geometry stage only."""

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
from eb_box_migration import project_stored_box
from q1_interface import FrozenQ1V5OOF
from q3_constraints import audit_concentrations
from constrained_migration import stored_from_concentrations
from run_e204_constrained_migration import TargetGeometry, _direct_vectors
from run_e215_strict_eb_migration import _candidate_concentrations, _path_status, validate_geometry_outputs


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = Path("/home/mary/mathorcup")
RESULT = ROOT / "03_结果/Q3_风格生成"
CONFIG = Path(__file__).resolve().parents[1] / "configs/E216_box_migration.yaml"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    required = {
        "experiment": "E216", "scope": "full_data_postselection_design_not_outer_validation",
        "source_style": "Pils/Lager", "target_styles": ["Tripel", "Kriek", "Lambic"],
        "scenario": "C1", "mean_kappa": 15.0, "covariance_kappa": 30.0,
        "direct_raw_lower_bound": 0.0, "log10_lower_bound": None,
        "kkt_tolerance": 1e-9, "path_lambdas": [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0],
        "source_count": 11, "target_pair_count": 33,
        "minimum_relative_target_gain": 0.01, "sensory_guard_margin": 0.01,
        "sensory_repeat_quantile": 0.9, "minimum_support_repeat_fraction": 0.9,
        "primary_selection_rule": "minimum_change", "auditor_used_during_search": False,
        "q1_retrained": False, "nnls_refit": False, "gpu_used": False,
        "kaggle_used_for_training": False, "chemical_safety_certified": False,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"E216 preregistered setting changed: {key}")
    return True


def registered_bounds(source_values, source_known, scales):
    scales = np.asarray(scales, dtype=str)
    known = np.asarray(source_known, dtype=bool)
    source_values = np.asarray(source_values, dtype=float)
    if source_values.shape != known.shape or known.shape != scales.shape or not np.isin(scales, ["raw", "log10"]).all():
        raise ValueError("aligned registered direct-defect metadata required")
    lower = np.where(scales == "raw", 0.0, -np.inf)
    upper = np.full(len(scales), np.inf)
    upper[known] = stored_from_concentrations(source_values[known], scales[known])
    return lower, upper


def run():
    started = time.perf_counter()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    RESULT.mkdir(parents=True, exist_ok=True)
    q1 = FrozenQ1V5OOF(CANONICAL)
    attachment = ROOT / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"
    source_path = RESULT / "q3_source_defects_E200.csv"
    defect_path = RESULT / "q3_defect_registry_E200.csv"
    contract_path = RESULT / "q3_field_contract_E200.csv"
    e215_manifest = RESULT / "q3_run_manifest_E215_geometry.json"
    e215_endpoints = RESULT / "q3_geometry_endpoints_E215.csv"
    inputs = [CONFIG, Path(__file__), Path(__file__).with_name("eb_box_migration.py"),
              Path(__file__).with_name("eb_migration.py"),
              Path(__file__).with_name("run_e215_strict_eb_migration.py"),
              Path(__file__).with_name("q3_constraints.py"), attachment, source_path,
              defect_path, contract_path, e215_manifest, e215_endpoints]
    input_hashes = {str(path.resolve()): sha(path) for path in inputs}
    manifest_path = RESULT / "q3_run_manifest_E216_geometry.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("input_sha256") != input_hashes or previous.get("q1_final_oof_sha256") != q1.signature:
            raise ValueError("E216 checkpoint input identity changed")
        for name, digest in previous["output_sha256"].items():
            if sha(RESULT / name) != digest:
                raise ValueError("E216 checkpoint output changed")
        print("E216 geometry checkpoint verified without rewriting", flush=True)
        return previous
    old_manifest = json.loads(e215_manifest.read_text(encoding="utf-8"))
    if old_manifest.get("status") != "complete" or old_manifest.get("q1_final_oof_sha256") != q1.signature:
        raise ValueError("E215 audited geometry baseline required")
    chemistry = pd.read_csv(attachment)
    features = [c for c in chemistry if c not in ["beer", "beer_id", "tasting_category_fine"]]
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
        raise ValueError("registered source count changed")
    geometries = {style: TargetGeometry(x[styles == style], global_mean, global_scale) for style in config["target_styles"]}
    endpoint_rows, path_rows, spectra_rows = [], [], []
    for pos in source_positions:
        beer = int(chemistry.beer.iloc[pos])
        source = x[pos]
        _, direct_fields, _, direct_scales, source_values, known, thresholds = _direct_vectors(
            beer, source_table, defect_table, field_contract
        )
        indices = np.array([features.index(name) for name in direct_fields], dtype=int)
        lower, upper = registered_bounds(source_values, known, direct_scales)
        for style in config["target_styles"]:
            anchor = model.anchor(source, config["source_style"], style)
            endpoint, dual = project_stored_box(model, anchor, style, indices, lower, upper,
                                                 tolerance=config["kkt_tolerance"])
            trajectory = migration_path(source, endpoint, config["path_lambdas"], include_identity=True)
            geometry = geometries[style]
            source_distance = float(geometry.distance(source)[0])
            target_block = x[styles == style]
            target_lower, target_upper = target_block.min(axis=0), target_block.max(axis=0)
            for path_index, (lam, candidate) in enumerate(zip([0.0] + config["path_lambdas"], trajectory)):
                concentrations = _candidate_concentrations(candidate[indices], direct_scales)
                direct_audit = audit_concentrations(source_values, concentrations, known, "C1", threshold=thresholds)
                n_failed = int((direct_audit.passed == False).fillna(False).sum())
                n_unknown = int(direct_audit.passed.isna().sum())
                distance = float(geometry.distance(candidate)[0])
                gain = float((source_distance - distance) / max(source_distance, 1e-12))
                outside = (candidate < target_lower) | (candidate > target_upper)
                row = {"beer": beer, "source_style": config["source_style"], "target_style": style,
                    "path_index": path_index, "lambda": float(lam), "source_identity": bool(lam == 0.0),
                    "source_target_distance": source_distance, "candidate_target_distance": distance,
                    "relative_target_gain": gain, "target_gain": bool(gain >= config["minimum_relative_target_gain"]),
                    "source_change_rms": float(np.sqrt(np.mean(((candidate - source) / global_scale) ** 2))),
                    "target_support_outside_cell_fraction": float(outside.mean()),
                    "n_direct_known": int(known.sum()), "n_direct_unknown": n_unknown,
                    "n_direct_failed": n_failed, "status": _path_status(n_failed, n_unknown, gain >= config["minimum_relative_target_gain"]),
                    "physical_safety_certified": False, "scope": config["scope"]}
                path_rows.append(row)
                spectra_rows.append({"beer": beer, "target_style": style, "path_index": path_index,
                                     "lambda": float(lam), **dict(zip(features, candidate))})
                if lam == 1.0:
                    endpoint_rows.append({**row, "unconstrained_anchor_target_distance": float(geometry.distance(anchor)[0]),
                        "projection_change_rms": float(np.sqrt(np.mean(((endpoint - anchor) / global_scale) ** 2))),
                        "n_active_lower_bounds": int((dual < -1e-9).sum()),
                        "n_active_upper_bounds": int((dual > 1e-9).sum())})
            if not np.array_equal(trajectory[0], source):
                raise ValueError("E216 lambda=0 source spectrum changed")
    endpoints = pd.DataFrame(endpoint_rows)
    paths = pd.DataFrame(path_rows)
    spectra = pd.DataFrame(spectra_rows)
    validate_geometry_outputs(endpoints, paths, config)
    old = pd.read_csv(e215_endpoints)
    comparison = endpoints.merge(old[["beer", "target_style", "relative_target_gain", "source_change_rms", "n_direct_failed", "target_support_outside_cell_fraction"]],
        on=["beer", "target_style"], validate="one_to_one", suffixes=("_new", "_e215"))
    if len(comparison) != 33:
        raise ValueError("E215/E216 paired source-target coverage incomplete")
    comparison["relative_target_gain_delta"] = comparison.relative_target_gain_new - comparison.relative_target_gain_e215
    outputs = []
    for name, frame in [("q3_geometry_endpoints_E216.csv", endpoints),
                        ("q3_geometry_paths_E216.csv", paths),
                        ("q3_geometry_spectra_E216.csv.gz", spectra),
                        ("q3_geometry_vs_E215_E216.csv", comparison)]:
        path = RESULT / name
        frame.to_csv(path, index=False, compression="gzip" if name.endswith(".gz") else None)
        outputs.append(path)
    for path, digest in input_hashes.items():
        if sha(path) != digest:
            raise ValueError("E216 frozen input changed during run")
    n_failed = int(paths.n_direct_failed.gt(0).sum())
    manifest = {"experiment_id": "E216", "status": "complete" if n_failed == 0 else "blocked_direct_constraint",
        "scope": config["scope"], "n_source_target_pairs": 33, "n_nonzero_paths": 231,
        "n_paths_including_source": 264, "n_direct_failed_path_rows": n_failed,
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
