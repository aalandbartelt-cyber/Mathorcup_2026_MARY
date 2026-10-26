"""Independent replay of E215's fixed post-selection geometry checkpoint."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import yaml

from q1_interface import FrozenQ1V5OOF
from q3_constraints import audit_concentrations
from run_e204_constrained_migration import TargetGeometry, _direct_vectors
from run_e215_strict_eb_migration import (
    CANONICAL, CONFIG, RESULT, ROOT, _candidate_concentrations,
    _path_status, sha, validate_config, validate_geometry_outputs,
)


def run():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    manifest_path = RESULT / "q3_run_manifest_E215_geometry.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E215" or manifest.get("status") != "complete" or manifest.get("n_source_target_pairs") != 33:
        raise ValueError("complete E215 geometry checkpoint required")
    for path, digest in manifest["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"frozen E215 input changed: {path}")
    for name, digest in manifest["output_sha256"].items():
        if sha(RESULT / name) != digest:
            raise ValueError(f"E215 output changed: {name}")
    if FrozenQ1V5OOF(CANONICAL).signature != manifest["q1_final_oof_sha256"]:
        raise ValueError("Q1 final model interface changed")
    chemistry = pd.read_csv(ROOT / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv")
    features = [c for c in chemistry if c not in ["beer", "beer_id", "tasting_category_fine"]]
    x = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(chemistry[features].to_numpy(float))
    styles = chemistry.tasting_category_fine.to_numpy(str)
    mean = x.mean(axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    geometries = {s: TargetGeometry(x[styles == s], mean, scale) for s in config["target_styles"]}
    positions = {int(beer): pos for pos, beer in enumerate(chemistry.beer)}
    source_table = pd.read_csv(RESULT / "q3_source_defects_E200.csv")
    defect_table = pd.read_csv(RESULT / "q3_defect_registry_E200.csv")
    contract = pd.read_csv(RESULT / "q3_field_contract_E200.csv")
    endpoints = pd.read_csv(RESULT / "q3_geometry_endpoints_E215.csv")
    paths = pd.read_csv(RESULT / "q3_geometry_paths_E215.csv")
    spectra = pd.read_csv(RESULT / "q3_geometry_spectra_E215.csv.gz")
    validate_geometry_outputs(endpoints, paths, config)
    joined = paths.merge(spectra, on=["beer", "target_style", "path_index", "lambda"], validate="one_to_one")
    if len(joined) != 264:
        raise ValueError("path/spectrum coverage incomplete")
    max_path_error = max_distance_error = 0.0
    invalid_raw_direct_rows = 0
    for (beer, style), group in joined.groupby(["beer", "target_style"], sort=True):
        pos = positions[int(beer)]
        source = x[pos]
        trajectory = group.sort_values("path_index")
        if not np.array_equal(trajectory.iloc[0][features].to_numpy(float), source):
            raise ValueError("lambda=0 source identity changed")
        endpoint = trajectory.iloc[-1][features].to_numpy(float)
        _, direct_fields, _, direct_scales, source_values, known, thresholds = _direct_vectors(
            beer, source_table, defect_table, contract
        )
        indices = [features.index(name) for name in direct_fields]
        geometry = geometries[style]
        source_distance = float(geometry.distance(source)[0])
        for _, row in trajectory.iterrows():
            candidate = row[features].to_numpy(float)
            lam = float(row["lambda"])
            expected = (1.0 - lam) * source + lam * endpoint
            max_path_error = max(max_path_error, float(np.max(np.abs(candidate - expected))))
            if max_path_error > 1e-8:
                raise ValueError("saved E215 spectrum differs from fixed convex path")
            concentrations = _candidate_concentrations(candidate[indices], direct_scales)
            audit = audit_concentrations(source_values, concentrations, known, "C1", threshold=thresholds)
            n_failed = int((audit.passed == False).fillna(False).sum())
            n_unknown = int(audit.passed.isna().sum())
            if n_failed != int(row.n_direct_failed) or n_unknown != int(row.n_direct_unknown):
                raise ValueError("saved direct-defect status differs from independent audit")
            if audit.status.eq("invalid_candidate").any():
                invalid_raw_direct_rows += 1
            distance = float(geometry.distance(candidate)[0])
            gain = float((source_distance - distance) / max(source_distance, 1e-12))
            max_distance_error = max(max_distance_error, abs(distance - float(row.candidate_target_distance)))
            if abs(gain - float(row.relative_target_gain)) > 1e-9:
                raise ValueError("saved target gain differs from replay")
            if row.status != _path_status(n_failed, n_unknown, gain >= config["minimum_relative_target_gain"]):
                raise ValueError("saved status differs from registered rule")
    report = {"experiment_id": "E215", "stage": "geometry_only", "status": "complete",
        "n_source_target_pairs": len(endpoints), "n_path_rows": len(paths),
        "max_spectrum_path_error": max_path_error, "max_target_distance_error": max_distance_error,
        "n_endpoint_direct_failed": int(endpoints.n_direct_failed.gt(0).sum()),
        "n_endpoint_all_direct_known": int(endpoints.n_direct_unknown.eq(0).sum()),
        "n_invalid_raw_direct_path_rows": invalid_raw_direct_rows,
        "mean_target_support_outside_cell_fraction": float(endpoints.target_support_outside_cell_fraction.mean()),
        "n_endpoint_gain_above_one_percent": int(endpoints.target_gain.sum()),
        "q1_retrained": False, "nnls_refit": False, "sensory_proxy_used": False,
        "physical_safety_certified": False,
        "decision": "geometry_only_not_eligible_for_paper_or_sensory_lock_until_nonnegative_direct_constraints_added",
        "input_manifest_sha256": sha(manifest_path)}
    output = RESULT / "q3_e215_geometry_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
