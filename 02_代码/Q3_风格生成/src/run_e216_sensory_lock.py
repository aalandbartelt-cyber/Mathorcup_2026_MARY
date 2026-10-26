"""Lock E216 spectra using only the frozen Q1 final NNLS, before auditor access."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from run_e216_box_migration import CANONICAL, CONFIG, RESULT, ROOT, sha, validate_config
from sensory_audit import load_read_only_nnls, migration_model_input, support_audit
from sensory_search import predict_single_target, select_candidates

Q2 = CANONICAL / ".worktrees/q2-evidence-opt"
Q2_SRC = Q2 / "02_代码/Q2_关键化合物/src"
Q2_RESULT = Q2 / "03_结果/Q2_关键化合物"
sys.path.insert(0, str(Q2_SRC))
from q1_final_adapter import FrozenQ1FinalModel


def validate_search_geometry(paths):
    expected_lambdas = [0.0, .05, .1, .2, .35, .5, .75, 1.0]
    if (len(paths) != 264 or paths[["beer", "target_style", "path_index"]].duplicated().any()
        or set(paths.target_style) != {"Tripel", "Kriek", "Lambic"}
        or paths.beer.nunique() != 11 or paths.n_direct_failed.ne(0).any()):
        raise ValueError("E216 complete zero-direct-failure geometry required")
    for _, group in paths.groupby(["beer", "target_style"]):
        group = group.sort_values("path_index")
        if (len(group) != 8 or group.path_index.tolist() != list(range(8))
            or not np.allclose(group["lambda"].to_numpy(float), expected_lambdas, rtol=0, atol=1e-12)):
            raise ValueError("E216 fixed 8-point path changed")
    return True


def choose_locked_path(group):
    selections, feasible = select_candidates(
        group.relative_target_gain, group.source_change_rms,
        group.primary_mean_delta, group.primary_repeat_q90_delta,
        group.support_repeat_fraction,
        minimum_gain=.01, guard_margin=.01,
    )
    return selections["minimum_change"], feasible


def frozen_inputs():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    manifest_path = RESULT / "q3_run_manifest_E216_geometry.json"
    audit_path = RESULT / "q3_e216_geometry_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or audit.get("status") != "complete"
        or audit.get("n_invalid_direct_rows") != 0 or audit.get("n_endpoint_direct_failed") != 0
        or audit.get("input_manifest_sha256") != sha(manifest_path)):
        raise ValueError("audited E216 geometry checkpoint required")
    for name, digest in manifest["output_sha256"].items():
        if sha(RESULT / name) != digest:
            raise ValueError(f"E216 geometry output changed: {name}")
    final = FrozenQ1FinalModel(CANONICAL)
    if final.registry["oof_sha256"] != manifest["q1_final_oof_sha256"]:
        raise ValueError("Q1 frozen OOF changed after E216 geometry")
    e115_path = Q2_RESULT / "q2_run_manifest_E115_final_nnls.json"
    e115 = json.loads(e115_path.read_text(encoding="utf-8"))
    if (e115.get("status") != "complete" or e115.get("n_outer_folds") != 30
        or e115.get("adapter_signature") != final.signature
        or e115.get("q1_retrained") is not False or e115.get("nnls_refit") is not False):
        raise ValueError("frozen Q1 predictor cache identity changed")
    chemistry_path = CANONICAL / "data/2026初赛/2026年MathorCup大数据挑战赛-赛道A初赛/附件/1.csv"
    chemistry = pd.read_csv(chemistry_path)
    x = chemistry.loc[:, final.feature_names]
    if (not np.array_equal(chemistry.beer.to_numpy(), final.arrays["beer"])
        or not np.array_equal(chemistry.tasting_category_fine.to_numpy(str), final.arrays["style"].astype(str))
        or not np.allclose(x.to_numpy(float), final.reference_x, rtol=0, atol=1e-12, equal_nan=True)):
        raise ValueError("E216 chemistry differs from frozen Q1 model input")
    paths = pd.read_csv(RESULT / "q3_geometry_paths_E216.csv")
    spectra = pd.read_csv(RESULT / "q3_geometry_spectra_E216.csv.gz")
    validate_search_geometry(paths)
    if (len(spectra) != 264 or spectra[["beer", "target_style", "path_index"]].duplicated().any()
        or set(final.feature_names) - set(spectra.columns)):
        raise ValueError("E216 spectra coverage or frozen feature order changed")
    for path, digest in manifest["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"E216 geometry input changed: {path}")
    input_files = [CONFIG, Path(__file__), Path(__file__).with_name("sensory_search.py"),
                   Path(__file__).with_name("sensory_audit.py"), manifest_path, audit_path,
                   chemistry_path, e115_path, final.registry_path, final.oof_path,
                   final.parameter_path, final.config_path, Q2_SRC / "q1_final_adapter.py"]
    hashes = {str(path.resolve()): sha(path) for path in input_files}
    hashes.update({str((RESULT / name).resolve()): digest for name, digest in manifest["output_sha256"].items()})
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return config, final, x, paths, spectra, hashes, signature


def run():
    lock_path = RESULT / "q3_candidate_lock_E216.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for name, digest in lock["output_sha256"].items():
            if sha(RESULT / name) != digest:
                raise ValueError("locked E216 output changed")
        print("E216 sensory candidates already locked; no reselection", flush=True)
        return lock
    config, final, x, paths, spectra, hashes, signature = frozen_inputs()
    features = list(final.feature_names)
    target_index = final.target_codes.index("stale_hops")
    source_positions = np.flatnonzero(final.arrays["style"].astype(str) == config["source_style"])
    source_beers = [int(final.arrays["beer"][pos]) for pos in source_positions]
    cache_dir = Q2_RESULT / "q2_nnls_cache_E115"
    checkpoint = RESULT / "q3_sensory_lock_checkpoints_E216"
    checkpoint.mkdir(exist_ok=True)
    all_diagnostics, all_selected, all_spectra = [], [], []
    cache_hashes = {}
    max_reconstruction_error = 0.0
    for source_number, pos in enumerate(source_positions, 1):
        beer = int(final.arrays["beer"][pos])
        base = checkpoint / f"beer_{beer}"
        base.mkdir(exist_ok=True)
        meta_path = base / "manifest.json"
        selected_path = base / "selected.csv"
        spectra_path = base / "spectra.csv.gz"
        diagnostics_path = base / "diagnostics.csv.gz"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("signature") != signature:
                raise ValueError("E216 source checkpoint input identity changed")
            for name, digest in meta["output_sha256"].items():
                if sha(base / name) != digest:
                    raise ValueError("E216 source checkpoint changed")
            all_selected.append(pd.read_csv(selected_path))
            all_spectra.append(pd.read_csv(spectra_path))
            all_diagnostics.append(pd.read_csv(diagnostics_path))
            cache_hashes.update(meta["model_cache_sha256"])
            max_reconstruction_error = max(max_reconstruction_error, meta["single_target_max_abs_error"])
            print(f"E216 sensory {source_number}/11 source={beer} cached", flush=True)
            continue
        group = paths[paths.beer == beer].loc[lambda frame: frame.path_index > 0].copy()
        candidate = spectra[spectra.beer == beer].loc[lambda frame: frame.path_index > 0].copy()
        group = group.sort_values(["target_style", "path_index"]).reset_index(drop=True)
        candidate = candidate.sort_values(["target_style", "path_index"]).reset_index(drop=True)
        if (len(group) != 21 or len(candidate) != 21
            or not group[["beer", "target_style", "path_index"]].equals(candidate[["beer", "target_style", "path_index"]])):
            raise ValueError("E216 source path/spectrum identity mismatch")
        query = migration_model_input(candidate[features], np.repeat(pos, len(candidate)), x)
        deltas, supports, local_hashes = [], [], {}
        source_error = 0.0
        for repeat in range(10):
            fold = int(final.arrays["fold_assignment"][repeat, pos])
            mask = final.arrays["fold_assignment"][repeat] == fold
            valid = np.flatnonzero(mask)
            train = np.flatnonzero(~mask)
            model, info = load_read_only_nnls(final, cache_dir, repeat, fold, x.iloc[valid])
            reconstruction = predict_single_target(model, x.iloc[valid], target_index)
            error = float(np.max(np.abs(reconstruction - final.arrays["prediction"][repeat, valid, target_index])))
            if error > 1e-10:
                raise ValueError("E216 single-target NNLS differs from frozen OOF")
            source_error = max(source_error, error)
            local_hashes[f"R{repeat:02d}F{fold}"] = info["model_sha256"]
            deltas.append(predict_single_target(model, query, target_index)
                          - float(final.arrays["prediction"][repeat, pos, target_index]))
            supports.append(support_audit(x.iloc[train], query)["supported"])
            del model
            gc.collect()
        deltas = np.stack(deltas)
        supports = np.stack(supports)
        group["primary_mean_delta"] = deltas.mean(axis=0)
        group["primary_repeat_q90_delta"] = np.quantile(deltas, config["sensory_repeat_quantile"], axis=0)
        group["support_repeat_fraction"] = supports.mean(axis=0)
        group["primary_feasible"] = False
        chosen, chosen_spectra = [], []
        for style in config["target_styles"]:
            indices = group.index[group.target_style == style].to_numpy()
            subset = group.loc[indices]
            selected_local, feasible = choose_locked_path(subset)
            group.loc[indices, "primary_feasible"] = feasible
            row = {"selection_id": f"{beer}_{style}_minimum_change", "beer": beer,
                   "source_style": config["source_style"], "target_style": style,
                   "scenario": "C1", "selection_rule": config["primary_selection_rule"],
                   "candidate_available": selected_local is not None,
                   "n_primary_feasible": int(feasible.sum()), "n_search_points": len(subset),
                   "auditor_accessed": False, "physical_safety_certified": False}
            if selected_local is None:
                row["search_status"] = "rejected_no_primary_feasible_candidate"
            else:
                index = int(indices[selected_local])
                row.update(group.loc[index].drop(labels=["primary_feasible"]).to_dict())
                row["search_status"] = "locked_primary_candidate"
                if row["n_direct_failed"] != 0:
                    raise ValueError("locked candidate violates known direct constraint")
                chosen_spectra.append({"selection_id": row["selection_id"],
                                       **candidate.loc[index, features].to_dict()})
            chosen.append(row)
        selected = pd.DataFrame(chosen)
        selected_spectra = pd.DataFrame(chosen_spectra, columns=["selection_id"] + features)
        group.to_csv(diagnostics_path, index=False, compression="gzip")
        selected.to_csv(selected_path, index=False)
        selected_spectra.to_csv(spectra_path, index=False, compression="gzip")
        meta = {"signature": signature, "beer": beer,
                "single_target_max_abs_error": source_error,
                "model_cache_sha256": local_hashes,
                "output_sha256": {p.name: sha(p) for p in [selected_path, spectra_path, diagnostics_path]}}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_diagnostics.append(group)
        all_selected.append(selected)
        all_spectra.append(selected_spectra)
        cache_hashes.update(local_hashes)
        max_reconstruction_error = max(max_reconstruction_error, source_error)
        print(f"E216 sensory {source_number}/11 source={beer} feasible={int(group.primary_feasible.sum())} locked={len(selected_spectra)}/3", flush=True)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    locked = pd.concat(all_spectra, ignore_index=True)
    if (len(diagnostics) != 231 or len(selected) != 33 or selected[["beer", "target_style"]].duplicated().any()
        or len(locked) != int(selected.candidate_available.sum())
        or set(locked.selection_id) != set(selected.loc[selected.candidate_available, "selection_id"])):
        raise ValueError("E216 complete pair denominator or locked spectra changed")
    outputs = []
    for name, table in [("q3_search_diagnostics_E216.csv.gz", diagnostics),
                        ("q3_search_selection_E216.csv", selected),
                        ("q3_locked_spectra_E216.csv.gz", locked)]:
        path = RESULT / name
        table.to_csv(path, index=False, compression="gzip" if name.endswith(".gz") else None)
        outputs.append(path)
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError("E216 sensory input changed during lock")
    if len(set(source_beers)) != 11:
        raise ValueError("E216 source count changed")
    lock = {"experiment_id": "E216", "status": "locked_before_model_family_audit",
            "scope": config["scope"], "n_source_target_pairs": 33,
            "n_search_points": 231, "n_locked_spectra": len(locked),
            "n_primary_feasible_paths": int(diagnostics.primary_feasible.sum()),
            "primary_single_target_max_abs_error": max_reconstruction_error,
            "selection_rule": config["primary_selection_rule"],
            "auditor_accessed": False, "q1_retrained": False, "nnls_refit": False,
            "gpu_used": False, "physical_safety_certified": False,
            "signature": signature, "input_sha256": hashes,
            "model_cache_sha256": cache_hashes,
            "checkpoint_metadata_sha256": {str(p.relative_to(checkpoint)): sha(p)
                                            for p in checkpoint.glob("beer_*/manifest.json")},
            "output_sha256": {p.name: sha(p) for p in outputs}}
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in lock.items() if not k.endswith("sha256")}, ensure_ascii=False), flush=True)
    return lock


if __name__ == "__main__":
    run()
