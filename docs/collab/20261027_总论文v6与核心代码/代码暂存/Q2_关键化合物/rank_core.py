# Source logical name: support_aware_rank_fusion; SHA256: 84ff0f46cf8fdf4577fbfff6cf3eb8deaab496d9ea5575f61eaff45ba79f2dce
# Source logical name: leave_style_rank_aggregation; SHA256: 1b8a9d049a2136ab85e438a9e7ebafe7d77e43b27e88b6b41b9cf4aca06f0b12
# Source logical name: four_evidence_connection; SHA256: 4c011e4d4a6dde6b52b97c812daaa6d48abfd22f0abd74e5e5af60045d029543
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


def _as_lens(
    frame: pd.DataFrame, rank_column: str, method: str
) -> pd.DataFrame:
    result = frame[["target_code", "feature_code", rank_column]].copy()
    result = result.rename(columns={rank_column: "aggregate_rank"})
    result.insert(0, "method", method)
    return result
