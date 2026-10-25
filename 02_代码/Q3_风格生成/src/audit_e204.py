"""Completeness, feasibility and scope audit for completed E204 migration."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "03_结果/Q3_风格生成"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    manifest_path = RESULT / "q3_run_manifest_E204.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("scope") != "full_data_postselection_design_not_outer_validation":
        raise ValueError("complete post-selection E204 manifest required")
    for name, digest in manifest["output_sha256"].items():
        if sha(RESULT / name) != digest:
            raise ValueError(f"E204 output hash mismatch: {name}")
    summary = pd.read_csv(RESULT / "q3_migration_summary_E204.csv")
    spectra = pd.read_csv(RESULT / "q3_migration_spectra_E204.csv.gz")
    direct = pd.read_csv(RESULT / "q3_migration_direct_audit_E204.csv")
    pool = pd.read_csv(RESULT / "q3_target_pool_E204.csv.gz")
    if len(summary) != 891 or int(summary.candidate_available.sum()) != 844:
        raise ValueError("E204 candidate count changed")
    if len(spectra) != 844 or len(direct) != 844 * 4 or len(pool) != 6000:
        raise ValueError("E204 spectrum, direct-audit or target-pool count changed")
    available = summary[summary.candidate_available].copy()
    unavailable = summary[~summary.candidate_available].copy()
    if not unavailable.status.eq("rejected_no_structural_candidate").all() or not unavailable.strategy.eq("structural_nearest").all():
        raise ValueError("unavailable candidates must be explicit structural rejections")
    if summary.status.eq("eligible_statistical_candidate").any() or summary.physical_safety_certified.eq(True).any():
        raise ValueError("E204 cannot certify candidates before the sensory proxy audit")
    unchanged = available[available.strategy == "source_unchanged"]
    if not np.allclose(unchanged.source_change_rms, 0.0) or unchanged.target_gain.any():
        raise ValueError("zero migration was counted as a target gain")
    projected = direct[direct.strategy.isin(["defect_only_repair", "coordinate_projection", "structural_nearest", "constrained_path"])]
    if projected.status.eq("violation").any():
        raise ValueError("projected/structural candidate has a known direct-constraint violation")
    if not (available[available.strategy == "constrained_path"]["lambda"] > 0).all():
        raise ValueError("zero lambda leaked into the migration path")
    audit_summary = {
        "status": "complete",
        "scope": manifest["scope"],
        "n_summary_rows": len(summary),
        "n_available_candidates": len(available),
        "n_structural_rejections": len(unavailable),
        "n_spectra": len(spectra),
        "n_direct_checks": len(direct),
        "n_target_pool": len(pool),
        "n_pareto": int(available.pareto_front.sum()),
        "status_counts": {str(key): int(value) for key, value in summary.status.value_counts().items()},
        "c1_path_target_gain_rate": float(
            available[(available.scenario == "C1") & (available.strategy == "constrained_path")].target_gain.mean()
        ),
        "c1_path_mean_relative_target_gain": float(
            available[(available.scenario == "C1") & (available.strategy == "constrained_path")].relative_target_gain.mean()
        ),
        "proxy_pending_candidates": int(summary.status.eq("needs_proxy_verification").sum()),
        "physical_safety_certified": False,
        "sensory_profiles_generated": False,
    }
    summary_path = RESULT / "q3_e204_audit_summary.json"
    summary_path.write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "status": "complete",
        "input_manifest_sha256": sha(manifest_path),
        "source_sha256": sha(Path(__file__)),
        "output_sha256": {summary_path.name: sha(summary_path)},
        "invariants": {
            "all_projected_known_constraints_nonviolating": True,
            "zero_migration_not_target_gain": True,
            "unknown_not_counted_as_pass": True,
            "no_final_candidate_before_proxy_audit": True,
        },
    }
    audit_path = RESULT / "q3_e204_audit_manifest.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit, "summary": audit_summary}, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    run()
