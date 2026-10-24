"""Audit Q1 second-round candidates against the frozen E003 gate."""

from __future__ import annotations

from collections.abc import Mapping


HARD_THRESHOLDS = {
    "r2": 0.2359,
    "mae": 0.3422,
    "spearman": 0.4260,
    "qwk_proxy": 0.4026,
    "acc_pm1_proxy": 0.5734,
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
