"""Deterministic aggregation helpers for the final Q2 robustness fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_leave_one_style_ranks(
    rankings: pd.DataFrame,
    *,
    expected_styles: int,
    rrf_k: float = 60.0,
) -> pd.DataFrame:
    """Aggregate complete leave-one-style ranks into one rank per target/feature."""
    required = {"omitted_style", "target_code", "feature_code", "consensus_rank"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"rankings miss columns: {sorted(missing)}")
    if int(expected_styles) < 2 or float(rrf_k) <= 0:
        raise ValueError("expected_styles must be >=2 and rrf_k must be positive")
    table = rankings[list(required)].copy()
    if table["omitted_style"].nunique() != int(expected_styles):
        raise ValueError("observed style count does not match expected_styles")
    coverage = table.groupby(["target_code", "feature_code"])[
        "omitted_style"
    ].nunique()
    if not coverage.eq(int(expected_styles)).all():
        raise ValueError("every target-feature pair must cover every omitted style")
    table["is_top3"] = table["consensus_rank"].le(3)
    table["is_top10"] = table["consensus_rank"].le(10)
    table["rrf_component"] = 1.0 / (float(rrf_k) + table["consensus_rank"])
    result = (
        table.groupby(["target_code", "feature_code"], sort=False)
        .agg(
            top3_survival_frequency=("is_top3", "mean"),
            top10_survival_frequency=("is_top10", "mean"),
            median_rank=("consensus_rank", "median"),
            rank_q25=("consensus_rank", lambda values: float(values.quantile(0.25))),
            rank_q75=("consensus_rank", lambda values: float(values.quantile(0.75))),
            mean_rrf=("rrf_component", "mean"),
        )
        .reset_index()
    )
    result["rank_iqr"] = result["rank_q75"] - result["rank_q25"]
    result = result.sort_values(
        [
            "target_code",
            "top3_survival_frequency",
            "top10_survival_frequency",
            "mean_rrf",
            "median_rank",
            "feature_code",
        ],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["aggregate_rank"] = result.groupby("target_code").cumcount() + 1
    return result
