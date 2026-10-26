"""Replay locked E217 spectra and frozen one-time audit without model reselection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from audit_e217_local_pair_search import verify_committed_lock
from constrained_migration import project_direct_constraints
from run_e204_constrained_migration import _direct_vectors
from run_e217_local_pair_search import RESULT, ROOT, frozen_search_inputs, sha


def reconstruct_locked_spectrum(source, a, b, lam, direct_indices, source_values,
                                source_known, thresholds, scales):
    source, a, b = (np.asarray(value, float) for value in (source, a, b))
    if (source.ndim != 1 or a.shape != source.shape or b.shape != source.shape
        or not np.isfinite(source).all() or not np.isfinite(a).all()
        or not np.isfinite(b).all() or not np.isfinite(lam) or not 0 < lam <= 1):
        raise ValueError("finite aligned source/anchors and nonzero lambda required")
    anchor = (a + b) / 2
    candidate = (1 - lam) * source + lam * anchor
    candidate[direct_indices], _ = project_direct_constraints(
        candidate[direct_indices], source_values, source_known, thresholds, scales, "C1")
    return candidate


def run():
    lock_path = RESULT / "q3_candidate_lock_E217.json"
    lock = verify_committed_lock(ROOT, lock_path)
    config, final, x_raw, pool, hashes, signature = frozen_search_inputs()
    if lock["signature"] != signature or lock["input_sha256"] != hashes:
        raise ValueError("E217 frozen inputs changed before replay")
    manifest_path = RESULT / "q3_run_manifest_E217.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete_once_only_audit"
        or manifest.get("candidate_lock_sha256") != sha(lock_path)
        or manifest.get("no_auditor_based_reselection") is not True):
        raise ValueError("completed once-only E217 audit required")
    for name, digest in manifest["output_sha256"].items():
        if sha(RESULT / name) != digest:
            raise ValueError("E217 audit output changed")
    selected = pd.read_csv(RESULT / "q3_search_selection_E217.csv")
    spectra = pd.read_csv(RESULT / "q3_locked_spectra_E217.csv.gz").set_index("selection_id")
    outcome = pd.read_csv(RESULT / "q3_once_audit_E217.csv")
    changes = pd.read_csv(RESULT / "q3_proxy_changes_by_repeat_E217.csv")
    direct = pd.read_csv(RESULT / "q3_direct_sensitivity_E217.csv")
    if (len(selected) != 33 or len(outcome) != 33
        or len(spectra) != manifest["n_locked_spectra"]
        or len(changes) != len(spectra) * 10 or len(direct) != len(spectra) * 12):
        raise ValueError("E217 selection, repeat or direct-defect denominator changed")
    features = list(final.feature_names)
    x_imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(x_raw)
    beers = {int(beer): position for position, beer in enumerate(final.arrays["beer"])}
    source_table = pd.read_csv(RESULT / "q3_source_defects_E200.csv")
    defect_table = pd.read_csv(RESULT / "q3_defect_registry_E200.csv")
    field_table = pd.read_csv(RESULT / "q3_field_contract_E200.csv")
    max_error = 0.
    for _, row in selected.iterrows():
        if not row.candidate_available:
            continue
        block = pool[pool["style"] == row.target_style].set_index("sample_id")
        a = block.loc[int(row.anchor_a_id), features].to_numpy(float)
        b = block.loc[int(row.anchor_b_id), features].to_numpy(float)
        source = x_imputed[beers[int(row.beer)]]
        _, direct_fields, _, scales, source_values, known, thresholds = _direct_vectors(
            int(row.beer), source_table, defect_table, field_table)
        indices = [features.index(field) for field in direct_fields]
        rebuilt = reconstruct_locked_spectrum(source, a, b, float(row["lambda"]),
            indices, source_values, known, thresholds, scales)
        error = float(np.max(np.abs(rebuilt - spectra.loc[row.selection_id, features].to_numpy(float))))
        if error > 1e-10:
            raise ValueError("E217 locked spectrum differs from its frozen anchor pair")
        max_error = max(max_error, error)
    joined = outcome.merge(selected[["selection_id", "beer", "target_style", "candidate_available"]],
                           on="selection_id", validate="one_to_one", suffixes=("", "_lock"))
    for column in ["beer", "target_style", "candidate_available"]:
        if not joined[column].equals(joined[column + "_lock"]):
            raise ValueError("E217 once-only audit changed a locked identity")
    if (outcome.loc[outcome.eligible, "n_direct_unknown"].fillna(1).ne(0).any()
        or outcome.physical_safety_certified.any()
        or int(outcome.eligible.sum()) != manifest["n_primary_eligible_pairs"]
        or bool(outcome.exploratory_new_lambic.any()) != manifest["new_lambic_exploratory_candidate"]):
        raise ValueError("E217 audit promoted unknown chemistry or changed its decision")
    report = {"experiment_id": "E217", "status": "complete", "n_source_target_pairs": 33,
              "n_paths": lock["n_search_points"], "n_locked_spectra_replayed": len(spectra),
              "max_locked_spectrum_replay_error": max_error,
              "n_primary_eligible_pairs": int(outcome.eligible.sum()),
              "n_unknown_marked_eligible": 0,
              "new_lambic_exploratory_candidate": bool(outcome.exploratory_new_lambic.any()),
              "no_auditor_based_reselection": True,
              "physical_safety_certified": False,
              "candidate_lock_sha256": sha(lock_path),
              "once_audit_manifest_sha256": sha(manifest_path)}
    path = RESULT / "q3_e217_replay_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
