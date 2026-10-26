"""Frozen-NNLS-only E217 search; lock before independent model-family audit."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

from local_pair_migration import nearest_pairs, fixed_paths
from constrained_migration import project_direct_constraints
from q3_constraints import audit_concentrations
from run_e204_constrained_migration import TargetGeometry, _direct_vectors, _candidate_direct
from run_e216_sensory_lock import CANONICAL, Q2_RESULT, frozen_inputs
from sensory_audit import sha, load_read_only_nnls, migration_model_input, support_audit
from sensory_search import predict_single_target


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "03_结果/Q3_风格生成"
CONFIG = Path(__file__).resolve().parents[1] / "configs/E217_local_pair_migration.yaml"


def validate_config(config):
    required = {
        "experiment": "E217", "question": "Q3", "status": "registered_before_run",
        "scope": "full_data_postselection_exploration_not_outer_validation",
        "source_style": "Pils/Lager", "target_styles": ["Tripel", "Kriek", "Lambic"],
        "scenario": "C1", "target_pool": "frozen_E204_style_copula_shrunk",
        "nearest_anchors": 24, "pair_weight": .5,
        "path_lambdas": [.05, .1, .2, .35, .5, .75, 1.],
        "source_count": 11, "target_pair_count": 33, "candidate_count": 63756,
        "minimum_relative_target_gain": .01, "sensory_guard_margin": .01,
        "sensory_repeat_quantile": .9, "minimum_support_repeat_fraction": .9,
        "primary_selection_rule": "minimum_change", "auditor_used_during_search": False,
        "q1_retrained": False, "nnls_refit": False, "gpu_used": False,
        "kaggle_used_for_training": False, "chemical_safety_certified": False,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"E217 preregistered setting changed: {key}")
    return True


def validate_ledger(table, *, n_sources, target_styles, pairs_per_group, lambdas):
    if (len(table) != n_sources * len(target_styles) * pairs_per_group * len(lambdas)
        or table.beer.nunique() != n_sources
        or set(table.target_style) != set(target_styles)
        or table[["beer", "target_style", "pair_index", "path_index"]].duplicated().any()):
        raise ValueError("E217 candidate identity/count changed")
    for _, group in table.groupby(["beer", "target_style"]):
        if len(group) != pairs_per_group * len(lambdas):
            raise ValueError("E217 pair candidate count changed")
        for pair, paths in group.groupby("pair_index"):
            paths = paths.sort_values("path_index")
            if (pair not in range(pairs_per_group) or len(paths) != len(lambdas)
                or paths.path_index.tolist() != list(range(len(lambdas)))
                or not np.allclose(paths["lambda"].to_numpy(float), lambdas, rtol=0, atol=1e-12)):
                raise ValueError("E217 frozen path order changed")
    return True


def select_primary(group):
    numeric = group.numeric_valid.to_numpy(bool)
    gain = group.relative_target_gain.to_numpy(float)
    change = group.source_change_rms.to_numpy(float)
    support = group.support_repeat_fraction.to_numpy(float)
    mean = group.primary_mean_delta.to_numpy(float)
    q90 = group.primary_repeat_q90_delta.to_numpy(float)
    feasible = (numeric & np.isfinite(gain) & np.isfinite(change)
                & np.isfinite(support) & np.isfinite(mean) & np.isfinite(q90)
                & (gain >= .01) & (mean <= -.01) & (q90 <= -.01) & (support >= .9))
    positions = np.flatnonzero(feasible)
    if not len(positions):
        return None, feasible
    order = np.lexsort((group["lambda"].to_numpy(float)[positions],
                        group.pair_index.to_numpy(int)[positions],
                        -gain[positions], change[positions]))
    return int(positions[order[0]]), feasible


def frozen_search_inputs():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    _, final, x, _, _, base_hashes, _ = frozen_inputs()
    e204_manifest_path = RESULT / "q3_run_manifest_E204.json"
    e204 = json.loads(e204_manifest_path.read_text(encoding="utf-8"))
    pool_path = RESULT / "q3_target_pool_E204.csv.gz"
    if (e204.get("status") != "complete"
        or e204["output_sha256"].get(pool_path.name) != sha(pool_path)):
        raise ValueError("audited frozen E204 proposal pool required")
    pool = pd.read_csv(pool_path)
    features = list(final.feature_names)
    if (len(pool) != 6000 or len(features) != 231
        or not set(features).issubset(pool.columns)
        or set(pool["style"]) != set(config["target_styles"])):
        raise ValueError("E204 target pool shape or Q1 feature order changed")
    for style in config["target_styles"]:
        block = pool[pool["style"] == style]
        if len(block) != 2000 or block.sample_id.duplicated().any():
            raise ValueError("E204 target style pool changed")
    files = [CONFIG, Path(__file__), Path(__file__).with_name("local_pair_migration.py"),
             Path(__file__).with_name("constrained_migration.py"),
             Path(__file__).with_name("sensory_audit.py"),
             Path(__file__).with_name("sensory_search.py"),
             e204_manifest_path, pool_path,
             RESULT / "q3_source_defects_E200.csv",
             RESULT / "q3_defect_registry_E200.csv",
             RESULT / "q3_field_contract_E200.csv"]
    hashes = dict(base_hashes)
    hashes.update({str(path.resolve()): sha(path) for path in files})
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return config, final, x, pool, hashes, signature


def source_proposals(beer, position, config, final, x_raw, x_imputed, pool,
                     source_table, defect_table, field_table, geometries, scale):
    source = x_imputed[position]
    features = list(final.feature_names)
    _, direct_fields, _, direct_scales, source_values, known, thresholds = _direct_vectors(
        beer, source_table, defect_table, field_table)
    direct_indices = [features.index(name) for name in direct_fields]
    blocks, descriptions = [], []
    for style in config["target_styles"]:
        target_pool = pool[pool["style"] == style]
        pair_ids, anchors = nearest_pairs(
            target_pool[features].to_numpy(float), source, scale,
            target_pool.sample_id.to_numpy(int), k=config["nearest_anchors"])
        values, anchor_index, lambdas = fixed_paths(source, anchors, config["path_lambdas"])
        numeric_valid = np.ones(len(values), dtype=bool)
        for index, row in enumerate(values):
            try:
                row[direct_indices], _ = project_direct_constraints(
                    row[direct_indices], source_values, known, thresholds, direct_scales, "C1")
            except ValueError:
                numeric_valid[index] = False
        source_distance = float(geometries[style].distance(source)[0])
        target_distance = geometries[style].distance(values)
        description = pd.DataFrame({
            "beer": beer, "source_style": config["source_style"], "target_style": style,
            "pair_index": anchor_index, "path_index": np.tile(np.arange(len(config["path_lambdas"])), len(anchors)),
            "anchor_a_id": pair_ids[anchor_index, 0], "anchor_b_id": pair_ids[anchor_index, 1],
            "lambda": lambdas, "numeric_valid": numeric_valid,
            "source_target_distance": source_distance, "candidate_target_distance": target_distance,
            "relative_target_gain": (source_distance - target_distance) / max(source_distance, 1e-12),
            "source_change_rms": np.sqrt(np.mean(((values - source) / scale) ** 2, axis=1)),
            "n_direct_known": int(known.sum()), "n_direct_unknown": int((~known).sum()),
        })
        blocks.append(values)
        descriptions.append(description)
    values = np.vstack(blocks)
    description = pd.concat(descriptions, ignore_index=True)
    if (len(values) != 5796 or len(description) != 5796
        or not np.array_equal(source, x_imputed[position])):
        raise ValueError("E217 source proposals or source identity changed")
    query = migration_model_input(values, np.repeat(position, len(values)), x_raw)
    return description, values, query, (direct_fields, direct_scales, source_values, known, thresholds)


def run():
    lock_path = RESULT / "q3_candidate_lock_E217.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for name, digest in lock["output_sha256"].items():
            if sha(RESULT / name) != digest:
                raise ValueError("locked E217 output changed")
        print("E217 candidates already locked; no reselection", flush=True)
        return lock
    config, final, x_raw, pool, hashes, signature = frozen_search_inputs()
    features = list(final.feature_names)
    target_index = final.target_codes.index("stale_hops")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_imputed = imputer.fit_transform(x_raw)
    styles = final.arrays["style"].astype(str)
    source_positions = np.flatnonzero(styles == config["source_style"])
    if len(source_positions) != config["source_count"]:
        raise ValueError("E217 frozen source count changed")
    mean = x_imputed.mean(axis=0)
    scale = np.std(x_imputed, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    geometries = {style: TargetGeometry(x_imputed[styles == style], mean, scale)
                  for style in config["target_styles"]}
    source_table = pd.read_csv(RESULT / "q3_source_defects_E200.csv")
    defect_table = pd.read_csv(RESULT / "q3_defect_registry_E200.csv")
    field_table = pd.read_csv(RESULT / "q3_field_contract_E200.csv")
    checkpoint = RESULT / "q3_search_checkpoints_E217"
    checkpoint.mkdir(exist_ok=True)
    summaries, spectra, diagnostics = [], [], []
    max_reconstruction_error, cache_hashes = 0., {}
    for number, position in enumerate(source_positions, 1):
        beer = int(final.arrays["beer"][position])
        folder = checkpoint / f"beer_{beer}"
        folder.mkdir(exist_ok=True)
        meta_path = folder / "manifest.json"
        summary_path = folder / "selected.csv"
        spectrum_path = folder / "spectra.csv.gz"
        diagnostic_path = folder / "diagnostics.csv.gz"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("signature") != signature:
                raise ValueError("E217 checkpoint signature changed")
            for name, digest in meta["output_sha256"].items():
                if sha(folder / name) != digest:
                    raise ValueError("E217 checkpoint output changed")
            summaries.append(pd.read_csv(summary_path))
            spectra.append(pd.read_csv(spectrum_path))
            diagnostics.append(pd.read_csv(diagnostic_path))
            cache_hashes.update(meta["model_cache_sha256"])
            max_reconstruction_error = max(max_reconstruction_error, meta["single_target_max_abs_error"])
            print(f"E217 {number}/11 source={beer} cached", flush=True)
            continue
        description, values, query, direct = source_proposals(
            beer, position, config, final, x_raw, x_imputed, pool,
            source_table, defect_table, field_table, geometries, scale)
        _, direct_scales, source_values, known, thresholds = direct
        supports, outside = [], []
        for repeat in range(10):
            fold = int(final.arrays["fold_assignment"][repeat, position])
            train = np.flatnonzero(final.arrays["fold_assignment"][repeat] != fold)
            support = support_audit(x_raw.iloc[train], query)
            supports.append(support["supported"])
            outside.append(support["outside_range_fraction"])
        supports = np.stack(supports)
        description["support_repeat_fraction"] = supports.mean(axis=0)
        description["mean_training_range_outside_fraction"] = np.stack(outside).mean(axis=0)
        preeligible = (description.numeric_valid.to_numpy(bool)
                       & (description.relative_target_gain.to_numpy(float) >= .01)
                       & (description.support_repeat_fraction.to_numpy(float) >= .9))
        selected_indices = np.flatnonzero(preeligible)
        mean_delta = np.full(len(description), np.nan)
        q90_delta = np.full(len(description), np.nan)
        model_hashes, local_error = {}, 0.
        if len(selected_indices):
            prediction_deltas = []
            selected_query = query[selected_indices]
            for repeat in range(10):
                fold = int(final.arrays["fold_assignment"][repeat, position])
                valid = np.flatnonzero(final.arrays["fold_assignment"][repeat] == fold)
                model, info = load_read_only_nnls(
                    final, Q2_RESULT / "q2_nnls_cache_E115", repeat, fold, x_raw.iloc[valid])
                error = float(np.max(np.abs(
                    predict_single_target(model, x_raw.iloc[valid], target_index)
                    - final.arrays["prediction"][repeat, valid, target_index])))
                if error > 1e-10:
                    raise ValueError("E217 single-target prediction differs from frozen NNLS OOF")
                local_error = max(local_error, error)
                model_hashes[f"R{repeat:02d}F{fold}"] = info["model_sha256"]
                prediction_deltas.append(
                    predict_single_target(model, selected_query, target_index)
                    - float(final.arrays["prediction"][repeat, position, target_index]))
                del model
                gc.collect()
            prediction_deltas = np.stack(prediction_deltas)
            mean_delta[selected_indices] = prediction_deltas.mean(axis=0)
            q90_delta[selected_indices] = np.quantile(
                prediction_deltas, config["sensory_repeat_quantile"], axis=0)
        description["primary_mean_delta"] = mean_delta
        description["primary_repeat_q90_delta"] = q90_delta
        description["primary_feasible"] = False
        chosen, chosen_spectra = [], []
        for style in config["target_styles"]:
            indices = description.index[description.target_style == style].to_numpy()
            subset = description.loc[indices].reset_index(drop=True)
            local, feasible = select_primary(subset)
            description.loc[indices, "primary_feasible"] = feasible
            record = {"selection_id": f"{beer}_{style}_minimum_change", "beer": beer,
                      "source_style": config["source_style"], "target_style": style,
                      "scenario": "C1", "selection_rule": "minimum_change",
                      "candidate_available": local is not None, "n_search_points": len(subset),
                      "n_primary_feasible": int(feasible.sum()),
                      "auditor_accessed": False, "physical_safety_certified": False}
            if local is None:
                record["search_status"] = "rejected_no_primary_feasible_candidate"
            else:
                index = int(indices[local])
                record.update(description.loc[index].drop(labels=["primary_feasible"]).to_dict())
                concentration = _candidate_direct(values[index],
                    [features.index(name) for name in direct[0]], direct_scales)
                check = audit_concentrations(source_values, concentration, known, "C1", threshold=thresholds)
                record["n_direct_failed"] = int(check.passed.eq(False).fillna(False).sum())
                record["n_direct_unknown"] = int(check.passed.isna().sum())
                if record["n_direct_failed"]:
                    raise ValueError("E217 selected known direct constraint failed")
                record["search_status"] = "locked_primary_candidate"
                chosen_spectra.append({"selection_id": record["selection_id"],
                                       **dict(zip(features, values[index]))})
            chosen.append(record)
        selected = pd.DataFrame(chosen)
        locked_spectra = pd.DataFrame(chosen_spectra, columns=["selection_id"] + features)
        description.to_csv(diagnostic_path, index=False, compression="gzip")
        selected.to_csv(summary_path, index=False)
        locked_spectra.to_csv(spectrum_path, index=False, compression="gzip")
        meta = {"signature": signature, "beer": beer,
                "n_paths": len(description), "n_primary_feasible": int(description.primary_feasible.sum()),
                "n_locked": len(locked_spectra), "single_target_max_abs_error": local_error,
                "model_cache_sha256": model_hashes,
                "output_sha256": {p.name: sha(p) for p in [summary_path, spectrum_path, diagnostic_path]}}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(selected)
        spectra.append(locked_spectra)
        diagnostics.append(description)
        cache_hashes.update(model_hashes)
        max_reconstruction_error = max(max_reconstruction_error, local_error)
        print(f"E217 {number}/11 source={beer} support_preeligible={len(selected_indices)} feasible={meta['n_primary_feasible']} locked={len(locked_spectra)}/3", flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    locked = pd.concat(spectra, ignore_index=True)
    diagnostic = pd.concat(diagnostics, ignore_index=True)
    validate_ledger(diagnostic, n_sources=11, target_styles=config["target_styles"],
                    pairs_per_group=276, lambdas=config["path_lambdas"])
    if (len(summary) != 33 or summary[["beer", "target_style"]].duplicated().any()
        or len(locked) != int(summary.candidate_available.sum())
        or set(locked.selection_id) != set(summary.loc[summary.candidate_available, "selection_id"])):
        raise ValueError("E217 locked pair denominator changed")
    outputs = []
    for name, table in [("q3_search_diagnostics_E217.csv.gz", diagnostic),
                        ("q3_search_selection_E217.csv", summary),
                        ("q3_locked_spectra_E217.csv.gz", locked)]:
        path = RESULT / name
        table.to_csv(path, index=False, compression="gzip" if name.endswith(".gz") else None)
        outputs.append(path)
    for name, digest in hashes.items():
        if sha(name) != digest:
            raise ValueError("E217 input changed during search")
    lock = {"experiment_id": "E217", "status": "locked_before_model_family_audit",
            "scope": config["scope"], "n_source_target_pairs": 33,
            "n_search_points": len(diagnostic), "n_locked_spectra": len(locked),
            "n_primary_feasible_paths": int(diagnostic.primary_feasible.sum()),
            "n_invalid_numeric_paths": int((~diagnostic.numeric_valid).sum()),
            "selection_rule": "minimum_change",
            "primary_single_target_max_abs_error": max_reconstruction_error,
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
