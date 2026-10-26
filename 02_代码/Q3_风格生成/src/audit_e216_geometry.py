"""Replay the E216 two-sided migration geometry without proxy access."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from q1_interface import FrozenQ1V5OOF
from q3_constraints import audit_concentrations
from run_e204_constrained_migration import TargetGeometry, _direct_vectors
from run_e215_strict_eb_migration import _candidate_concentrations, _path_status, validate_geometry_outputs
from run_e216_box_migration import CANONICAL, RESULT, ROOT, CONFIG, sha, validate_config


def run():
    import yaml

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    manifest_path = RESULT / "q3_run_manifest_E216_geometry.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E216" or manifest.get("status") != "complete" or manifest.get("n_direct_failed_path_rows") != 0:
        raise ValueError("complete zero-direct-failure E216 geometry checkpoint required")
    for path, digest in manifest["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"E216 input hash changed: {path}")
    for name, digest in manifest["output_sha256"].items():
        if sha(RESULT / name) != digest:
            raise ValueError(f"E216 output hash changed: {name}")
    if FrozenQ1V5OOF(CANONICAL).signature != manifest["q1_final_oof_sha256"]:
        raise ValueError("frozen Q1 interface changed")
    chemistry = pd.read_csv(ROOT / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv")
    features = [c for c in chemistry if c not in ["beer", "beer_id", "tasting_category_fine"]]
    x = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(chemistry[features].to_numpy(float))
    styles = chemistry.tasting_category_fine.to_numpy(str)
    mean = x.mean(axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    geometry = {style: TargetGeometry(x[styles == style], mean, scale) for style in config["target_styles"]}
    positions = {int(beer): pos for pos, beer in enumerate(chemistry.beer)}
    source_table = pd.read_csv(RESULT / "q3_source_defects_E200.csv")
    defect_table = pd.read_csv(RESULT / "q3_defect_registry_E200.csv")
    contract = pd.read_csv(RESULT / "q3_field_contract_E200.csv")
    endpoints = pd.read_csv(RESULT / "q3_geometry_endpoints_E216.csv")
    paths = pd.read_csv(RESULT / "q3_geometry_paths_E216.csv")
    spectra = pd.read_csv(RESULT / "q3_geometry_spectra_E216.csv.gz")
    validate_geometry_outputs(endpoints, paths, config)
    joined = paths.merge(spectra, on=["beer", "target_style", "path_index", "lambda"], validate="one_to_one")
    if len(joined) != 264:
        raise ValueError("E216 spectra/path coverage incomplete")
    max_path_error = 0.0
    max_distance_error = 0.0
    invalid_direct = 0
    for (beer, target), group in joined.groupby(["beer", "target_style"]):
        source = x[positions[int(beer)]]
        ordered = group.sort_values("path_index")
        if not np.array_equal(ordered.iloc[0][features].to_numpy(float), source):
            raise ValueError("E216 source identity changed")
        endpoint = ordered.iloc[-1][features].to_numpy(float)
        _, direct_fields, _, direct_scales, source_values, known, thresholds = _direct_vectors(
            beer, source_table, defect_table, contract
        )
        indices = [features.index(name) for name in direct_fields]
        source_distance = float(geometry[target].distance(source)[0])
        for _, row in ordered.iterrows():
            candidate = row[features].to_numpy(float)
            lam = float(row["lambda"])
            expected = (1.0 - lam) * source + lam * endpoint
            max_path_error = max(max_path_error, float(np.max(np.abs(expected - candidate))))
            if max_path_error > 1e-8:
                raise ValueError("E216 fixed path spectrum differs")
            concentrations = _candidate_concentrations(candidate[indices], direct_scales)
            direct_audit = audit_concentrations(source_values, concentrations, known, "C1", threshold=thresholds)
            failed = int((direct_audit.passed == False).fillna(False).sum())
            unknown = int(direct_audit.passed.isna().sum())
            invalid_direct += int(direct_audit.status.eq("invalid_candidate").any())
            if failed != int(row.n_direct_failed) or unknown != int(row.n_direct_unknown):
                raise ValueError("E216 direct-defect audit differs")
            distance = float(geometry[target].distance(candidate)[0])
            max_distance_error = max(max_distance_error, abs(distance - float(row.candidate_target_distance)))
            gain = float((source_distance - distance) / max(source_distance, 1e-12))
            if abs(gain - float(row.relative_target_gain)) > 1e-9 or row.status != _path_status(failed, unknown, gain >= .01):
                raise ValueError("E216 geometry or status differs")
    if invalid_direct or paths.n_direct_failed.gt(0).any():
        raise ValueError("E216 two-sided projection did not remove direct invalidity")
    complete_beers = source_table.groupby("beer").source_known.all().loc[lambda x: x].index
    report = {"experiment_id": "E216", "stage": "geometry_only", "status": "complete",
        "n_source_target_pairs": len(endpoints), "n_path_rows": len(paths),
        "max_spectrum_path_error": max_path_error, "max_target_distance_error": max_distance_error,
        "n_invalid_direct_rows": invalid_direct, "n_endpoint_direct_failed": int(endpoints.n_direct_failed.gt(0).sum()),
        "n_endpoint_all_direct_observed": int(endpoints.beer.isin(complete_beers).sum()),
        "n_endpoint_gain_above_one_percent": int(endpoints.target_gain.sum()),
        "mean_target_support_outside_cell_fraction": float(endpoints.target_support_outside_cell_fraction.mean()),
        "q1_retrained": False, "nnls_refit": False, "sensory_proxy_used": False,
        "physical_safety_certified": False,
        "decision": "numerically_valid_geometry_only_ready_for_registered_sensory_lock",
        "input_manifest_sha256": sha(manifest_path)}
    out = RESULT / "q3_e216_geometry_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
