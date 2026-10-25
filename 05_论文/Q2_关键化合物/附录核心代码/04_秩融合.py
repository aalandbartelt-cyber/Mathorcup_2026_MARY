"""Deterministic rank-consensus helpers for Q2稳健秩融合."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reciprocal_rank_consensus(
    rankings: pd.DataFrame,
    *,
    expected_methods: int,
    rrf_k: float = 60.0,
) -> pd.DataFrame:
    """Fuse complete method rankings with support-aware reciprocal rank fusion."""
    required = {"method", "target_code", "feature_code", "aggregate_rank"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"rankings miss columns: {sorted(missing)}")
    if int(expected_methods) < 2 or float(rrf_k) <= 0:
        raise ValueError("expected_methods must be >=2 and rrf_k must be positive")
    table = rankings[list(required)].copy()
    if table["method"].nunique() != int(expected_methods):
        raise ValueError("method count does not match expected_methods")
    duplicated = table.duplicated(["method", "target_code", "feature_code"])
    if duplicated.any():
        raise ValueError("each method-target-feature row must be unique")
    coverage = table.groupby(["target_code", "feature_code"])["method"].nunique()
    if not coverage.eq(int(expected_methods)).all():
        raise ValueError("every target-feature pair must cover every method")

    table["rrf_component"] = 1.0 / (float(rrf_k) + table["aggregate_rank"])
    table["is_top3"] = table["aggregate_rank"].le(3)
    table["is_top10"] = table["aggregate_rank"].le(10)
    result = (
        table.groupby(["target_code", "feature_code"], sort=False)
        .agg(
            rrf_score=("rrf_component", "sum"),
            method_top3_support=("is_top3", "sum"),
            method_top10_support=("is_top10", "sum"),
            best_method_rank=("aggregate_rank", "min"),
            mean_method_rank=("aggregate_rank", "mean"),
            median_method_rank=("aggregate_rank", "median"),
        )
        .reset_index()
    )
    result = result.sort_values(
        [
            "target_code",
            "method_top10_support",
            "method_top3_support",
            "rrf_score",
            "best_method_rank",
            "feature_code",
        ],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["consensus_rank"] = result.groupby("target_code").cumcount() + 1
    ranks = table.pivot(
        index=["target_code", "feature_code"],
        columns="method",
        values="aggregate_rank",
    ).reset_index()
    ranks.columns = [
        column if column in ("target_code", "feature_code") else f"rank__{column}"
        for column in ranks.columns
    ]
    result = result.merge(
        ranks, on=["target_code", "feature_code"], how="left", validate="one_to_one"
    )
    return result


def signed_direction_consensus(directions: list[str]) -> str:
    """Combine signed methods while allowing importance-only methods to abstain."""
    signed = [value for value in directions if value in {"positive", "negative"}]
    if not signed:
        return "unknown"
    if len(set(signed)) == 1:
        return signed[0]
    return "uncertain"
