"""Audit Q1 second-round candidates against the frozen E003 gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json

import pandas as pd

from data_contract import RESULT_DIR


HARD_THRESHOLDS = {
    "r2": 0.2359,
    "mae": 0.3422,
    "spearman": 0.4260,
    "qwk_proxy": 0.4026,
    "acc_pm1_proxy": 0.5734,
}
METRICS = ["r2", "mae", "spearman", "qwk_proxy", "acc_pm1_proxy"]


def screening_decision(
    candidate_means: Mapping[str, float], baseline_means: Mapping[str, float]
) -> dict:
    """Apply the pre-registered three-repeat promotion rule."""
    required = set(HARD_THRESHOLDS)
    missing = required - set(candidate_means)
    missing |= required - set(baseline_means)
    if missing:
        raise ValueError(f"missing screening metrics: {sorted(missing)}")
    improved = {
        "r2": float(candidate_means["r2"]) > float(baseline_means["r2"]),
        "mae": float(candidate_means["mae"]) < float(baseline_means["mae"]),
        "spearman": float(candidate_means["spearman"])
        > float(baseline_means["spearman"]),
        "qwk_proxy": float(candidate_means["qwk_proxy"])
        > float(baseline_means["qwk_proxy"]),
        "acc_pm1_proxy": float(candidate_means["acc_pm1_proxy"])
        > float(baseline_means["acc_pm1_proxy"]),
    }
    continuous_tolerance = {
        "r2": float(candidate_means["r2"])
        >= float(baseline_means["r2"]) - 0.002,
        "mae": float(candidate_means["mae"])
        <= float(baseline_means["mae"]) + 0.001,
    }
    promote = (
        sum(improved.values()) >= 4
        and (improved["spearman"] or improved["acc_pm1_proxy"])
        and all(continuous_tolerance.values())
    )
    return {
        "promote_to_full": bool(promote),
        "improved": improved,
        "improved_count": int(sum(improved.values())),
        "continuous_tolerance": continuous_tolerance,
    }


def promotion_decision(
    metric_means: Mapping[str, float],
    repeat_win_counts: Mapping[str, int],
    *,
    minimum_repeat_wins: int = 7,
) -> dict:
    """Apply the pre-registered five strict means and five 7/10 gates."""
    missing = set(HARD_THRESHOLDS) - set(metric_means)
    missing |= set(HARD_THRESHOLDS) - set(repeat_win_counts)
    if missing:
        raise ValueError(f"missing promotion metrics: {sorted(missing)}")
    mean_flags = {
        "r2": float(metric_means["r2"]) > HARD_THRESHOLDS["r2"],
        "mae": float(metric_means["mae"]) < HARD_THRESHOLDS["mae"],
        "spearman": float(metric_means["spearman"])
        > HARD_THRESHOLDS["spearman"],
        "qwk_proxy": float(metric_means["qwk_proxy"])
        > HARD_THRESHOLDS["qwk_proxy"],
        "acc_pm1_proxy": float(metric_means["acc_pm1_proxy"])
        > HARD_THRESHOLDS["acc_pm1_proxy"],
    }
    repeat_flags = {
        name: int(repeat_win_counts[name]) >= int(minimum_repeat_wins)
        for name in HARD_THRESHOLDS
    }
    return {
        "promoted": bool(all(mean_flags.values()) and all(repeat_flags.values())),
        "mean_threshold_flags": mean_flags,
        "repeat_win_flags": repeat_flags,
        "minimum_repeat_wins": int(minimum_repeat_wins),
    }


def _repeat_rows(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    selected = frame.loc[frame["evaluation"].str.startswith(prefix)].copy()
    if selected.empty:
        raise RuntimeError(f"no repeat metrics found for prefix {prefix}")
    selected["repeat"] = selected["evaluation"].str.rsplit("_", n=1).str[-1].astype(int)
    if selected["repeat"].duplicated().any():
        raise RuntimeError(f"duplicated repeat metrics for prefix {prefix}")
    return selected[["repeat", *METRICS]].sort_values("repeat").reset_index(drop=True)


def audit_e004_screening() -> tuple[pd.DataFrame, dict]:
    """Compare CPU/GPU E004 screening with E003 on repeats 0--2 only."""
    baseline_all = pd.read_csv(RESULT_DIR / "q1_extended_metrics_E003.csv")
    baseline_rows = _repeat_rows(baseline_all, "repeat_")
    candidates = [
        (
            "E004",
            "ordinal_hgb",
            RESULT_DIR / "q1_extended_metrics_E004_screening.csv",
            "pred_ordinal_hgb:repeat_",
        ),
        (
            "E004_GPU",
            "ordinal_xgb_cuda",
            RESULT_DIR / "q1_extended_metrics_E004_GPU_screening.csv",
            "pred_ordinal_xgb:repeat_",
        ),
    ]
    summary_rows = []
    repeat_frames = []
    decisions = {}
    for experiment_id, model, path, prefix in candidates:
        candidate_rows = _repeat_rows(pd.read_csv(path), prefix)
        repeat_ids = candidate_rows["repeat"].tolist()
        baseline = baseline_rows.loc[baseline_rows["repeat"].isin(repeat_ids)].copy()
        if baseline["repeat"].tolist() != repeat_ids:
            raise RuntimeError(f"E003 repeat IDs do not align with {experiment_id}")
        baseline_means = {name: float(baseline[name].mean()) for name in METRICS}
        candidate_means = {
            name: float(candidate_rows[name].mean()) for name in METRICS
        }
        decision = screening_decision(candidate_means, baseline_means)
        decisions[experiment_id] = {
            "model": model,
            "repeat_ids": repeat_ids,
            "baseline_means": baseline_means,
            "candidate_means": candidate_means,
            **decision,
        }
        summary_row = {
            "experiment_id": experiment_id,
            "model": model,
            "n_repeats": len(repeat_ids),
            "promote_to_full": decision["promote_to_full"],
            "improved_count": decision["improved_count"],
        }
        for name in METRICS:
            summary_row[f"baseline_{name}"] = baseline_means[name]
            summary_row[f"candidate_{name}"] = candidate_means[name]
            summary_row[f"delta_{name}"] = candidate_means[name] - baseline_means[name]
            summary_row[f"improved_{name}"] = decision["improved"][name]
        summary_rows.append(summary_row)
        paired = candidate_rows.merge(
            baseline, on="repeat", suffixes=("_candidate", "_baseline")
        )
        paired.insert(0, "model", model)
        paired.insert(0, "experiment_id", experiment_id)
        repeat_frames.append(paired)

    summary = pd.DataFrame(summary_rows)
    by_repeat = pd.concat(repeat_frames, ignore_index=True)
    summary_path = RESULT_DIR / "q1_screening_comparison_E004.csv"
    repeat_path = RESULT_DIR / "q1_screening_by_repeat_E004.csv"
    decision_path = RESULT_DIR / "q1_screening_decision_E004.json"
    summary.to_csv(summary_path, index=False)
    by_repeat.to_csv(repeat_path, index=False)
    payload = {
        "status": "screening_only",
        "full_promotion_requires": "10 repeats and all five hard gates plus 7/10 wins",
        "decisions": decisions,
        "outputs": [str(summary_path), str(repeat_path)],
    }
    decision_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["e004-screening"], default="e004-screening"
    )
    args = parser.parse_args()
    if args.mode == "e004-screening":
        summary, payload = audit_e004_screening()
        print(summary.to_string(index=False))
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
