"""Once-only independent family audit of the committed E217 local-pair lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from run_e217_local_pair_search import RESULT, ROOT, frozen_search_inputs, select_primary, validate_ledger
from run_e205_sensory_audit import CONFIG as E205_CONFIG, Q2_RESULT, load_inputs, transform_elastic
from run_e204_constrained_migration import _direct_vectors, _candidate_direct
from selection_models import (fit_fold_preprocessor, build_quadratic_penalty_rows,
                              fit_multi_target_with_penalty_rows)
from sensory_audit import (sha, load_read_only_nnls, migration_model_input,
                           support_audit, verify_archived_prediction, proxy_screen,
                           candidate_status)
from sensory_search import predict_single_target
from q3_constraints import audit_concentrations


def verify_committed_lock(root, lock_path):
    root, lock_path = Path(root), Path(lock_path)
    relative = str(lock_path.relative_to(root))
    try:
        committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=root,
                                            stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise ValueError("E217 candidate lock must be committed before audit") from exc
    if committed != lock_path.read_bytes():
        raise ValueError("E217 candidate lock changed after commit")
    lock = json.loads(committed)
    if (lock.get("status") != "locked_before_model_family_audit"
        or lock.get("auditor_accessed") is not False
        or lock.get("n_source_target_pairs") != 33
        or lock.get("n_search_points") != 63756):
        raise ValueError("E217 committed candidate lock protocol changed")
    for name, digest in lock["output_sha256"].items():
        path = lock_path.parent / name
        relative_output = str(path.relative_to(root))
        if sha(path) != digest:
            raise ValueError("E217 locked output changed after search")
        try:
            committed_output = subprocess.check_output(
                ["git", "show", "HEAD:" + relative_output], cwd=root, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            raise ValueError("E217 locked output must be committed before audit") from exc
        if hashlib.sha256(committed_output).hexdigest() != digest:
            raise ValueError("E217 committed output differs from locked output")
    return lock


def decide_exploratory(row):
    return bool(
        row["beer"] == 230 and row["target_style"] == "Lambic"
        and row["candidate_available"] and row["n_direct_failed"] == 0
        and row["n_direct_unknown"] == 0 and row["support_repeat_fraction"] >= .9
        and row["relative_target_gain"] >= .01
        and row["primary_mean_delta"] <= -.01
        and row["primary_repeat_q90_delta"] <= -.01
        and row["audit_mean_delta"] <= -.01
        and row["audit_repeat_q90_delta"] <= -.01
    )


def run():
    output_manifest = RESULT / "q3_run_manifest_E217.json"
    if output_manifest.exists():
        previous = json.loads(output_manifest.read_text(encoding="utf-8"))
        for name, digest in previous["output_sha256"].items():
            if sha(RESULT / name) != digest:
                raise ValueError("completed E217 audit output changed")
        print("E217 once-only audit already complete; no re-audit", flush=True)
        return previous
    lock_path = RESULT / "q3_candidate_lock_E217.json"
    lock = verify_committed_lock(ROOT, lock_path)
    config, final, x, pool, hashes, signature = frozen_search_inputs()
    if lock["signature"] != signature or lock["input_sha256"] != hashes:
        raise ValueError("E217 frozen search inputs changed before audit")
    diagnostics = pd.read_csv(RESULT / "q3_search_diagnostics_E217.csv.gz")
    selected = pd.read_csv(RESULT / "q3_search_selection_E217.csv")
    spectra = pd.read_csv(RESULT / "q3_locked_spectra_E217.csv.gz")
    validate_ledger(diagnostics, n_sources=11, target_styles=config["target_styles"],
                    pairs_per_group=276, lambdas=config["path_lambdas"])
    features = list(final.feature_names)
    if (len(selected) != 33 or selected[["beer", "target_style"]].duplicated().any()
        or spectra.selection_id.duplicated().any()
        or set(spectra.selection_id) != set(selected.loc[selected.candidate_available, "selection_id"])):
        raise ValueError("E217 complete locked set or 33-pair denominator changed")
    for (beer, style), group in diagnostics.groupby(["beer", "target_style"]):
        subset = group.sort_values(["pair_index", "path_index"]).reset_index(drop=True)
        index, feasible = select_primary(subset)
        row = selected.loc[selected.beer.eq(beer) & selected.target_style.eq(style)]
        if (len(row) != 1 or int(feasible.sum()) != int(row.iloc[0].n_primary_feasible)
            or bool(row.iloc[0].candidate_available) != (index is not None)):
            raise ValueError("E217 fixed primary-only choice changed")
        if index is not None:
            chosen = subset.iloc[index]
            for column in ["pair_index", "path_index", "anchor_a_id", "anchor_b_id", "lambda"]:
                if not np.isclose(float(row.iloc[0][column]), float(chosen[column]), rtol=0, atol=1e-12):
                    raise ValueError("E217 locked anchor or lambda was reselected")
    for name, digest in lock["checkpoint_metadata_sha256"].items():
        path = RESULT / "q3_search_checkpoints_E217" / name
        if sha(path) != digest:
            raise ValueError("E217 source checkpoint metadata changed")
        for output, output_digest in json.loads(path.read_text())["output_sha256"].items():
            if sha(path.parent / output) != output_digest:
                raise ValueError("E217 source checkpoint changed")
    data = load_inputs(__import__("yaml").safe_load(E205_CONFIG.read_text()))
    if data["final"].signature != final.signature or not np.array_equal(data["x"].to_numpy(), x.to_numpy(), equal_nan=True):
        raise ValueError("E217 auditor model input differs from frozen search input")
    codes = list(final.target_codes)
    j = codes.index("stale_hops")
    selected_available = selected[selected.candidate_available].reset_index(drop=True)
    metadata = spectra[["selection_id"]].merge(selected_available, on="selection_id", validate="one_to_one")
    n = len(metadata)
    positions = np.array([list(final.arrays["beer"]).index(int(beer)) for beer in metadata.beer], dtype=int)
    query = migration_model_input(spectra[features], positions, x) if n else np.empty((0, len(features)))
    primary = np.full((10, n), np.nan)
    audit = np.full((10, n), np.nan)
    supported = np.zeros((10, n), bool)
    reconstruction = []
    if n:
        for repeat in range(10):
            for fold in range(3):
                mask = final.arrays["fold_assignment"][repeat] == fold
                valid = np.flatnonzero(mask)
                train = np.flatnonzero(~mask)
                candidates = np.flatnonzero(mask[positions])
                if not len(candidates):
                    continue
                model, info = load_read_only_nnls(
                    final, Q2_RESULT / "q2_nnls_cache_E115", repeat, fold, x.iloc[valid])
                prediction = predict_single_target(model, query[candidates], j)
                primary[repeat, candidates] = prediction - final.arrays["prediction"][repeat, positions[candidates], j]
                processed = fit_fold_preprocessor(x.iloc[train], x.iloc[valid])
                parameters = data["parameters"].query("repeat==@repeat and fold==@fold").iloc[0]
                p = processed.x_train.shape[1]
                penalty = build_quadratic_penalty_rows(np.zeros((p, p)), n_samples=len(train),
                                                       lambda_l2=float(parameters.lambda_l2), lambda_graph=0.)
                estimator = fit_multi_target_with_penalty_rows(
                    processed.x_train, data["y"][train], penalty,
                    lambda_l1=float(parameters.lambda_l1))
                auditor_error = verify_archived_prediction(
                    estimator.predict(processed.x_validation), data["auditor_oof"][repeat, valid])
                candidate_prediction = estimator.predict(transform_elastic(processed, query[candidates]))[:, j]
                source_prediction = estimator.predict(
                    transform_elastic(processed, x.iloc[positions[candidates]]))[:, j]
                audit[repeat, candidates] = candidate_prediction - source_prediction
                supported[repeat, candidates] = support_audit(
                    x.iloc[train], query[candidates])["supported"]
                reconstruction.append({"repeat": repeat, "fold": fold,
                                       "primary_error": info["reconstruction_max_abs_error"],
                                       "auditor_archive_quantization_error": auditor_error})
        if not np.isfinite(primary).all() or not np.isfinite(audit).all():
            raise ValueError("E217 frozen model-family audit incomplete")
    reliability = pd.read_csv(RESULT / "q3_proxy_model_reliability_E205.csv")
    reliable = reliability[reliability.target_code.eq("stale_hops")].set_index("model").proxy_reliable
    source_table = pd.read_csv(RESULT / "q3_source_defects_E200.csv")
    defect_table = pd.read_csv(RESULT / "q3_defect_registry_E200.csv")
    field_table = pd.read_csv(RESULT / "q3_field_contract_E200.csv")
    checks, outcomes, changes = [], [], []
    for index, row in metadata.iterrows():
        proxy = proxy_screen(primary[:, index], audit[:, index],
                             bool(reliable.final_NNLS), bool(reliable.ElasticNet_auditor))
        support_fraction = float(supported[:, index].mean())
        if (abs(proxy["primary_mean_delta"] - row.primary_mean_delta) > 1e-10
            or abs(proxy["primary_repeat_q90_delta"] - row.primary_repeat_q90_delta) > 1e-10
            or abs(support_fraction - row.support_repeat_fraction) > 1e-12):
            raise ValueError("E217 locked primary scores or support were not reproducible")
        _, fields, defects, scales, source_values, known, thresholds = _direct_vectors(
            int(row.beer), source_table, defect_table, field_table)
        concentration = _candidate_direct(spectra.loc[index, features].to_numpy(float),
                                          [features.index(field) for field in fields], scales)
        n_failed = n_unknown = 0
        for scenario in ["C1", "C2", "C3"]:
            table = audit_concentrations(source_values, concentration, known, scenario, threshold=thresholds)
            for defect, check in zip(defects, table.to_dict("records")):
                checks.append({"selection_id": row.selection_id, "beer": int(row.beer),
                               "target_style": row.target_style, "defect": defect, **check})
            if scenario == "C1":
                n_failed = int(table.passed.eq(False).fillna(False).sum())
                n_unknown = int(table.passed.isna().sum())
        status = candidate_status(available=True, n_failed=n_failed, n_unknown=n_unknown,
                                  target_gain=row.relative_target_gain >= .01,
                                  proxy_status=proxy["proxy_status"],
                                  support_fraction=support_fraction)
        outcome = {"selection_id": row.selection_id, **proxy,
                   "n_direct_failed": n_failed, "n_direct_unknown": n_unknown,
                   "audit_support_repeat_fraction": support_fraction,
                   "audited_status": status}
        outcomes.append(outcome)
        for repeat in range(10):
            changes.append({"selection_id": row.selection_id, "repeat": repeat,
                            "primary_delta_z": primary[repeat, index],
                            "audit_delta_z": audit[repeat, index],
                            "supported": bool(supported[repeat, index])})
    outcome = selected.merge(pd.DataFrame(outcomes), on="selection_id", how="left",
                             validate="one_to_one", suffixes=("_search", "")) if n else selected.copy()
    if not n:
        for column in ["n_direct_failed", "n_direct_unknown", "audit_mean_delta", "audit_repeat_q90_delta",
                       "audit_support_repeat_fraction"]:
            outcome[column] = np.nan
    outcome.loc[~outcome.candidate_available, "audited_status"] = "rejected_no_primary_feasible_candidate"
    outcome["eligible"] = outcome.audited_status.eq("eligible_statistical_candidate")
    outcome["physical_safety_certified"] = False
    outcome["exploratory_new_lambic"] = False
    for index in outcome.index[outcome.candidate_available]:
        row = outcome.loc[index]
        outcome.loc[index, "exploratory_new_lambic"] = decide_exploratory(row)
    if len(outcome) != 33 or outcome.exploratory_new_lambic.sum() > 1:
        raise ValueError("E217 audit denominator or exploratory decision changed")
    output_files = []
    for name, table in [("q3_once_audit_E217.csv", outcome),
                        ("q3_direct_sensitivity_E217.csv", pd.DataFrame(checks)),
                        ("q3_proxy_changes_by_repeat_E217.csv", pd.DataFrame(changes)),
                        ("q3_predictor_reconstruction_E217.csv", pd.DataFrame(reconstruction))]:
        path = RESULT / name
        table.to_csv(path, index=False)
        output_files.append(path)
    old = pd.read_csv(RESULT / "q3_once_audit_E205B.csv")
    old_main = old[old.selection_rule.eq("minimum_change")]
    if len(old_main) != 33:
        raise ValueError("old E205B main comparison denominator changed")
    comparison = pd.DataFrame([
        {"target_style": style, "n_pairs": 11,
         "old_main_eligible": int(old_main.loc[old_main.target_style.eq(style), "eligible"].sum()),
         "new_main_eligible": int(outcome.loc[outcome.target_style.eq(style), "eligible"].sum()),
         "new_main_available": int(outcome.loc[outcome.target_style.eq(style), "candidate_available"].sum()),
         "new_main_unknown": int(outcome.loc[outcome.target_style.eq(style), "audited_status"].eq("needs_direct_verification").sum())}
        for style in config["target_styles"]])
    comparison_path = RESULT / "q3_search_comparison_E217.csv"
    comparison.to_csv(comparison_path, index=False)
    output_files.append(comparison_path)
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError("E217 frozen search input changed during auditor evaluation")
    manifest = {"experiment_id": "E217", "status": "complete_once_only_audit",
                "scope": config["scope"], "candidate_lock_sha256": sha(lock_path),
                "candidate_lock_commit": subprocess.check_output(
                    ["git", "log", "-1", "--format=%H", "--", str(lock_path.relative_to(ROOT))],
                    cwd=ROOT, text=True).strip(),
                "audit_source_sha256": sha(__file__),
                "n_source_target_pairs": 33, "n_search_points": 63756,
                "n_locked_spectra": n, "n_primary_eligible_pairs": int(outcome.eligible.sum()),
                "n_primary_eligible_source_beers": int(outcome.loc[outcome.eligible, "beer"].nunique()),
                "status_counts": {str(k): int(v) for k, v in outcome.audited_status.value_counts().items()},
                "new_lambic_exploratory_candidate": bool(outcome.exploratory_new_lambic.any()),
                "no_auditor_based_reselection": True, "q1_retrained": False,
                "nnls_refit": False, "gpu_used": False,
                "physical_safety_certified": False, "main_generator_replaced": False,
                "output_sha256": {path.name: sha(path) for path in output_files}}
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("sha256")},
                     ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    run()
