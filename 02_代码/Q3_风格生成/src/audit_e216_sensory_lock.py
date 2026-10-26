"""Independently verify the committed E216 primary-only lock and its full denominator."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from run_e216_sensory_lock import RESULT, ROOT, choose_locked_path, frozen_inputs, sha


def verify_committed_lock(root, path):
    root, path = Path(root), Path(path)
    relative = str(path.relative_to(root))
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=root)
    if committed != path.read_bytes():
        raise ValueError("E216 candidate lock must be committed unchanged before audit")
    lock = json.loads(committed)
    if (lock.get("status") != "locked_before_model_family_audit"
        or lock.get("auditor_accessed") is not False
        or lock.get("n_source_target_pairs") != 33
        or lock.get("n_search_points") != 231):
        raise ValueError("E216 primary-only lock protocol changed")
    for name, digest in lock["output_sha256"].items():
        file = path.parent / name
        if sha(file) != digest:
            raise ValueError(f"locked E216 output changed: {name}")
        subprocess.check_call(["git", "ls-files", "--error-unmatch", str(file.relative_to(root))],
                              cwd=root, stdout=subprocess.DEVNULL)
    return lock


def run():
    lock_path = RESULT / "q3_candidate_lock_E216.json"
    lock = verify_committed_lock(ROOT, lock_path)
    _, final, _, paths, spectra, hashes, signature = frozen_inputs()
    if lock["signature"] != signature or lock["input_sha256"] != hashes:
        raise ValueError("E216 search inputs changed after committed candidate lock")
    diagnostics = pd.read_csv(RESULT / "q3_search_diagnostics_E216.csv.gz")
    selected = pd.read_csv(RESULT / "q3_search_selection_E216.csv")
    locked = pd.read_csv(RESULT / "q3_locked_spectra_E216.csv.gz")
    if (len(diagnostics) != 231 or len(selected) != 33 or selected[["beer", "target_style"]].duplicated().any()
        or set(selected.beer) != set(paths.beer) or set(selected.target_style) != set(paths.target_style)
        or not locked.empty):
        raise ValueError("E216 complete 33-pair/231-path denominator or empty locked set changed")
    if (lock["n_locked_spectra"] != 0 or lock["n_primary_feasible_paths"] != 0
        or bool(selected.candidate_available.any()) or bool(diagnostics.primary_feasible.any())):
        raise ValueError("E216 candidate was silently promoted or removed")
    for (beer, style), group in diagnostics.groupby(["beer", "target_style"]):
        group = group.sort_values("path_index")
        choice, feasible = choose_locked_path(group)
        record = selected.loc[selected.beer.eq(beer) & selected.target_style.eq(style)]
        if (len(group) != 7 or len(record) != 1 or choice is not None
            or feasible.any() or int(record.iloc[0].n_primary_feasible) != 0):
            raise ValueError("E216 primary-only fixed rule no longer rejects this pair")
        original = paths.loc[paths.beer.eq(beer) & paths.target_style.eq(style) & paths.path_index.gt(0)]
        if not np.array_equal(group.path_index.to_numpy(), original.sort_values("path_index").path_index.to_numpy()):
            raise ValueError("E216 geometry/search path identities differ")
    geometry = pd.read_csv(RESULT / "q3_geometry_endpoints_E216.csv")
    all_known = int(geometry.n_direct_unknown.eq(0).sum())
    if all_known != 3:
        raise ValueError("known/unknown source defect denominator changed")
    gain = diagnostics.relative_target_gain >= .01
    mean = diagnostics.primary_mean_delta <= -.01
    q90 = diagnostics.primary_repeat_q90_delta <= -.01
    support = diagnostics.support_repeat_fraction >= .9
    if int((gain & mean & q90 & support).sum()) != 0:
        raise ValueError("E216 had a feasible path but none was locked")
    for name, digest in lock["checkpoint_metadata_sha256"].items():
        path = RESULT / "q3_sensory_lock_checkpoints_E216" / name
        if sha(path) != digest:
            raise ValueError("E216 checkpoint metadata changed")
        for output, output_digest in json.loads(path.read_text())["output_sha256"].items():
            if sha(path.parent / output) != output_digest:
                raise ValueError("E216 per-source checkpoint output changed")
    # The Q2 worktree is a sibling of this Q3 worktree. No cache is written here.
    cache_dir = ROOT.parent / "q2-evidence-opt" / "03_结果/Q2_关键化合物/q2_nnls_cache_E115"
    for key, digest in lock["model_cache_sha256"].items():
        if sha(cache_dir / f"{key}.joblib") != digest:
            raise ValueError("read-only frozen NNLS cache changed after E216 lock")
    report = {"experiment_id": "E216", "status": "complete_no_locked_candidate",
              "stage": "committed_primary_only_sensory_lock_audit",
              "n_source_target_pairs": 33, "n_nonzero_paths": 231,
              "n_locked_spectra": 0, "n_all_direct_fields_observed_pairs": all_known,
              "n_pairs_with_unknown_direct_fields": 33 - all_known,
              "n_paths_target_gain": int(gain.sum()),
              "n_paths_mean_proxy_margin": int(mean.sum()),
              "n_paths_repeat_q90_proxy_margin": int(q90.sum()),
              "n_paths_predictor_support": int(support.sum()),
              "n_paths_target_and_two_proxy_margins": int((gain & mean & q90).sum()),
              "n_paths_all_registered_conditions": 0,
              "single_target_max_abs_error": lock["primary_single_target_max_abs_error"],
              "auditor_called": False, "auditor_skip_reason": "no_locked_spectrum_to_audit",
              "q1_retrained": False, "nnls_refit": False,
              "no_auditor_based_reselection": True, "physical_safety_certified": False,
              "main_generator_replaced": False,
              "decision": "retain_frozen_style_copula_and_existing_migration_results",
              "candidate_lock_sha256": sha(lock_path),
              "candidate_lock_commit": subprocess.check_output(
                  ["git", "log", "-1", "--format=%H", "--", str(lock_path.relative_to(ROOT))],
                  cwd=ROOT, text=True).strip(),
              "frozen_q1_oof_sha256": final.registry["oof_sha256"]}
    path = RESULT / "q3_e216_sensory_lock_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
