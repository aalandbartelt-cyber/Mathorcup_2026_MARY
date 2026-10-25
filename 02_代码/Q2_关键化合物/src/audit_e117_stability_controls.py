"""Integrity and convergence audit for Q2 E117 CPSS/null controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "03_结果" / "Q2_关键化合物"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def topk_jaccard(left: pd.DataFrame, right: pd.DataFrame, k: int) -> list[float]:
    merged = left.merge(right, on=["target_code", "feature_code"], suffixes=("_pilot", "_full"))
    values = []
    for _, frame in merged.groupby("target_code", sort=True):
        pilot = set(frame.nlargest(k, "selection_probability_pilot")["feature_code"])
        full = set(frame.nlargest(k, "selection_probability_full")["feature_code"])
        values.append(len(pilot & full) / len(pilot | full))
    return values


def main() -> None:
    manifest_path = RESULT / "q2_run_manifest_E117.json"
    pilot_manifest_path = RESULT / "q2_run_manifest_E117_pilot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    selection = pd.read_csv(RESULT / "q2_cpss_selection_E117.csv")
    pilot = pd.read_csv(RESULT / "q2_cpss_selection_E117_pilot.csv")
    pair_jaccard = pd.read_csv(RESULT / "q2_cpss_pair_jaccard_E117.csv")
    null = pd.read_csv(RESULT / "q2_null_target_permutation_E117.csv.gz")
    shadow = pd.read_csv(RESULT / "q2_shadow_feature_audit_E117.csv")
    error = pd.read_csv(RESULT / "q2_error_control_E117.csv")
    alignment = pd.read_csv(RESULT / "q2_cpss_nnls_alignment_E117.csv.gz")
    splits = pd.read_csv(RESULT / "q2_cpss_split_indices_E117.csv.gz")

    split_ok = True
    for _, frame in splits.groupby("pair", sort=True):
        left = set(frame.loc[frame["half"].eq("left"), "row_index"])
        right = set(frame.loc[frame["half"].eq("right"), "row_index"])
        split_ok &= not (left & right) and left | right == set(range(250))

    actual_hashes = {
        Path(path).name: sha256(Path(path)) for path in manifest["outputs"]
    }
    output_hashes_match = actual_hashes == manifest["output_sha256"]
    confirmed = error.loc[error["is_confirmed_threshold"].eq(True)].copy()
    expected_pairs = {(target, feature) for target in selection["target_code"].unique() for feature in selection["feature_code"].unique()}
    observed_pairs = set(map(tuple, selection[["target_code", "feature_code"]].to_numpy()))
    checks = {
        "manifest_complete": manifest.get("status") == "complete" and manifest.get("stage") == "full",
        "pilot_complete": pilot_manifest.get("status") == "pilot_complete",
        "formal_100_pairs_200_trials": manifest.get("n_pairs") == 100
        and manifest.get("selection_trials_per_target") == 200,
        "dimensions_50_by_218": len(selection) == 10_900
        and selection["target_code"].nunique() == 50
        and selection["feature_code"].nunique() == 218
        and observed_pairs == expected_pairs,
        "selection_counts_consistent": np.allclose(
            selection["selection_count"] / selection["selection_trials"],
            selection["selection_probability"],
        ),
        "complementary_splits_valid": bool(split_ok) and len(splits) == 25_000,
        "pair_jaccard_complete": len(pair_jaccard) == 5_000
        and pair_jaccard.groupby("target_code")["pair"].nunique().eq(100).all(),
        "null_complete": len(null) == 10_900
        and null["target_code"].nunique() == 50
        and null["feature_code"].nunique() == 218,
        "shadow_complete": len(shadow) == 2_000
        and shadow.groupby("target_code").size().eq(40).all(),
        "error_grid_complete": len(error) == 200
        and error.groupby("target_code")["threshold"].nunique().eq(4).all(),
        "confirmed_fdp_within_limit": confirmed["empirical_fdp"].le(0.10).all(),
        "final_nnls_alignment_complete": len(alignment) == 10_900
        and alignment["nnls_fold_count"].eq(30).all()
        and not alignment[["nnls_median_delta_mae", "nnls_predictive_rank"]].isna().any().any(),
        "e115_hash_traceable": manifest.get("e115_conditional_sha256")
        == "77d75e16b00f5677ccea85e517a703ecce33e420d9e1b4c37e69358366f8160c",
        "upstream_gates_passed": manifest.get("upstream", {}).get("status") == "ready"
        and manifest.get("upstream", {}).get("e115_output_hashes_match") is True,
        "output_hashes_match": output_hashes_match,
    }
    comparison = pilot.merge(
        selection,
        on=["target_code", "feature_code"],
        suffixes=("_pilot", "_full"),
        validate="one_to_one",
    )
    top3 = topk_jaccard(pilot, selection, 3)
    top10 = topk_jaccard(pilot, selection, 10)
    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        "experiment_id": "E117",
        "audit_status": "pass" if not failed else "fail",
        "failed_checks": failed,
        "checks": checks,
        "counts": {
            "pairs": 100,
            "selection_rows": len(selection),
            "pair_jaccard_rows": len(pair_jaccard),
            "confirmed_targets": int(confirmed["target_code"].nunique()),
            "confirmed_relations": int(confirmed["real_selected"].sum()),
        },
        "formal_summary": {
            "mean_pairwise_jaccard": float(pair_jaccard["jaccard"].mean()),
            "median_pairwise_jaccard": float(pair_jaccard["jaccard"].median()),
            "real_selected_by_threshold": error.groupby("threshold")["real_selected"].sum().to_dict(),
            "null_selected_by_threshold": error.groupby("threshold")["null_selected"].sum().to_dict(),
            "shadow_real_selected": int(shadow["real_selected"].sum()),
            "shadow_selected": int(shadow["shadow_selected"].sum()),
        },
        "pilot_to_formal": {
            "spearman_all_probabilities": float(
                comparison[["selection_probability_pilot", "selection_probability_full"]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "mean_absolute_probability_change": float(
                (comparison["selection_probability_pilot"] - comparison["selection_probability_full"])
                .abs()
                .mean()
            ),
            "mean_top3_jaccard": float(np.mean(top3)),
            "median_top3_jaccard": float(np.median(top3)),
            "mean_top10_jaccard": float(np.mean(top10)),
        },
        "output_sha256": actual_hashes,
    }
    output = RESULT / "q2_audit_E117.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    if failed:
        raise SystemExit(f"E117 audit failed: {failed}")


if __name__ == "__main__":
    main()
