"""Fuse total, bootstrap, style-adjusted, and leave-style-out Q2 ranks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from q2_contract import RESULT_DIR
from rank_consensus import reciprocal_rank_consensus
from robust_consensus import aggregate_leave_one_style_ranks


EXPERIMENT_ID = "E111"
LENS_TOTAL = "E106_total_association"
LENS_BOOTSTRAP = "E107_bootstrap_stability"
LENS_STYLE = "E108_style_adjusted"
LENS_DELETE = "E108_leave_one_style_survival"


def _as_lens(
    frame: pd.DataFrame, rank_column: str, method: str
) -> pd.DataFrame:
    result = frame[["target_code", "feature_code", rank_column]].copy()
    result = result.rename(columns={rank_column: "aggregate_rank"})
    result.insert(0, "method", method)
    return result


def _top3_comparison(
    final: pd.DataFrame, alternate: pd.DataFrame, rank_column: str, label: str
) -> pd.DataFrame:
    rows = []
    for target in sorted(final["target_code"].unique()):
        left = set(
            final.loc[
                final["target_code"].eq(target) & final["consensus_rank"].le(3),
                "feature_code",
            ]
        )
        right = set(
            alternate.loc[
                alternate["target_code"].eq(target)
                & alternate[rank_column].le(3),
                "feature_code",
            ]
        )
        rows.append(
            {
                "lens": label,
                "target_code": target,
                "intersection": len(left & right),
                "union": len(left | right),
                "jaccard": len(left & right) / len(left | right),
                "exact_top3": left == right,
            }
        )
    return pd.DataFrame(rows)


def main(output_dir: Path = RESULT_DIR) -> None:
    output_dir = Path(output_dir)
    total = pd.read_csv(output_dir / "q2_consensus_rankings_E106.csv")
    bootstrap = pd.read_csv(output_dir / "q2_bootstrap_selection_E107.csv")
    adjusted = pd.read_csv(output_dir / "q2_style_adjusted_rankings_E108.csv")
    deletion_folds = pd.read_csv(output_dir / "q2_leave_one_style_rankings_E108.csv.gz")
    deletion = aggregate_leave_one_style_ranks(deletion_folds, expected_styles=22)

    lenses = pd.concat(
        [
            _as_lens(total, "consensus_rank", LENS_TOTAL),
            _as_lens(bootstrap, "bootstrap_rank", LENS_BOOTSTRAP),
            _as_lens(adjusted, "consensus_rank", LENS_STYLE),
            _as_lens(deletion, "aggregate_rank", LENS_DELETE),
        ],
        ignore_index=True,
    )
    consensus = reciprocal_rank_consensus(lenses, expected_methods=4, rrf_k=60)
    consensus = consensus.rename(
        columns={
            "method_top3_support": "lens_top3_support",
            "method_top10_support": "lens_top10_support",
        }
    )

    e106_columns = total[
        [
            "target_code",
            "feature_code",
            "consensus_rank",
            "method_top3_support",
            "method_top10_support",
            "direction_consensus",
            "statistical_consensus_tier",
        ]
    ].rename(
        columns={
            "consensus_rank": "total_association_rank",
            "method_top3_support": "base_method_top3_support",
            "method_top10_support": "base_method_top10_support",
            "statistical_consensus_tier": "base_statistical_tier",
        }
    )
    bootstrap_columns = bootstrap[
        [
            "target_code",
            "feature_code",
            "bootstrap_rank",
            "selection_frequency",
            "median_rank",
            "rank_iqr",
            "direction_consistency_when_selected",
        ]
    ].rename(
        columns={
            "selection_frequency": "bootstrap_selection_frequency",
            "median_rank": "bootstrap_median_rank",
            "rank_iqr": "bootstrap_rank_iqr",
        }
    )
    adjusted_columns = adjusted[
        [
            "target_code",
            "feature_code",
            "consensus_rank",
            "method_top3_support",
            "method_top10_support",
        ]
    ].rename(
        columns={
            "consensus_rank": "style_adjusted_rank",
            "method_top3_support": "style_method_top3_support",
            "method_top10_support": "style_method_top10_support",
        }
    )
    deletion_columns = deletion.rename(
        columns={
            "aggregate_rank": "leave_style_rank",
            "top3_survival_frequency": "leave_style_top3_frequency",
            "top10_survival_frequency": "leave_style_top10_frequency",
            "median_rank": "leave_style_median_rank",
            "rank_iqr": "leave_style_rank_iqr",
        }
    )
    result = (
        consensus.merge(e106_columns, on=["target_code", "feature_code"], validate="one_to_one")
        .merge(bootstrap_columns, on=["target_code", "feature_code"], validate="one_to_one")
        .merge(adjusted_columns, on=["target_code", "feature_code"], validate="one_to_one")
        .merge(
            deletion_columns[
                [
                    "target_code",
                    "feature_code",
                    "leave_style_rank",
                    "leave_style_top3_frequency",
                    "leave_style_top10_frequency",
                    "leave_style_median_rank",
                    "leave_style_rank_iqr",
                ]
            ],
            on=["target_code", "feature_code"],
            validate="one_to_one",
        )
    )

    metric = (
        pd.read_csv(output_dir / "q2_metrics_by_target_E105.csv")
        .groupby("target_code", sort=False)
        .agg(
            e105_mean_oof_r2=("r2", "mean"),
            e105_mean_oof_mae=("mae", "mean"),
            e105_mean_oof_spearman=("spearman", "mean"),
        )
        .reset_index()
    )
    candidates = pd.read_csv(output_dir / "q2_candidate_dictionary.csv")
    targets = pd.read_csv(output_dir / "q2_target_dictionary.csv")
    modules = pd.read_csv(output_dir / "q2_stable_module_membership_E107.csv")
    result = (
        result.merge(metric, on="target_code", validate="many_to_one")
        .merge(
            candidates[
                [
                    "feature_code",
                    "compound_name",
                    "chemical_name_short",
                    "unit",
                    "origin_paper",
                    "aging",
                    "off_flavors",
                    "threshold_mean",
                    "threshold_median",
                    "has_calibrated_values",
                    "has_sensory_threshold",
                ]
            ],
            on="feature_code",
            validate="many_to_one",
        )
        .merge(
            targets[["target_code", "target_name_en", "target_group", "direct_anchor"]],
            on="target_code",
            validate="many_to_one",
        )
        .merge(
            modules[["feature_code", "stable_module_id", "module_size"]],
            on="feature_code",
            validate="many_to_one",
        )
    )
    result["style_evidence"] = np.select(
        [result["style_adjusted_rank"].le(3), result["style_adjusted_rank"].le(10)],
        ["within_style_top3", "within_style_top10"],
        default="style_sensitive",
    )
    result["predictability_tier"] = np.select(
        [
            result["e105_mean_oof_r2"].ge(0.20),
            result["e105_mean_oof_r2"].ge(0.10),
        ],
        ["P1_supported", "P2_limited"],
        default="P3_weak",
    )
    strong = (
        result["lens_top10_support"].ge(3)
        & result["lens_top3_support"].ge(2)
        & result["bootstrap_selection_frequency"].ge(0.60)
        & result["leave_style_top3_frequency"].ge(0.60)
        & result["style_adjusted_rank"].le(10)
    )
    moderate = (
        result["lens_top10_support"].ge(2)
        & result["lens_top3_support"].ge(1)
        & result["bootstrap_selection_frequency"].ge(0.30)
        & result["leave_style_top3_frequency"].ge(0.30)
    )
    result["statistical_robustness_tier"] = np.select(
        [strong, moderate], ["R1_strong", "R2_moderate"], default="R3_exploratory"
    )
    result["direct_anchor_match"] = (
        result["direct_anchor"].notna()
        & result["feature_code"].eq(result["direct_anchor"])
    )
    result["attachment_annotation_count"] = (
        result[
            [
                "has_calibrated_values",
                "has_sensory_threshold",
            ]
        ]
        .astype(bool)
        .sum(axis=1)
        + result["aging"].notna().astype(int)
        + result["off_flavors"].notna().astype(int)
    )
    result = result.sort_values(
        ["target_code", "consensus_rank"], kind="mergesort"
    ).reset_index(drop=True)
    top3 = result.loc[result["consensus_rank"].le(3)].copy()

    comparisons = pd.concat(
        [
            _top3_comparison(result, total, "consensus_rank", LENS_TOTAL),
            _top3_comparison(result, bootstrap, "bootstrap_rank", LENS_BOOTSTRAP),
            _top3_comparison(result, adjusted, "consensus_rank", LENS_STYLE),
            _top3_comparison(result, deletion, "aggregate_rank", LENS_DELETE),
        ],
        ignore_index=True,
    )
    anchor_audit = targets.loc[targets["direct_anchor"].notna()].merge(
        result,
        left_on=["target_code", "direct_anchor"],
        right_on=["target_code", "feature_code"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_final"),
    )

    deletion.to_csv(output_dir / "q2_leave_one_style_all_candidates_E111.csv", index=False)
    result.to_csv(output_dir / "q2_robust_consensus_rankings_E111.csv", index=False)
    top3.to_csv(output_dir / "q2_final_top3_statistical_E111.csv", index=False)
    comparisons.to_csv(output_dir / "q2_final_lens_agreement_E111.csv", index=False)
    anchor_audit[
        [
            "target_code",
            "target_name_en",
            "direct_anchor",
            "consensus_rank",
            "statistical_robustness_tier",
            "bootstrap_selection_frequency",
            "style_adjusted_rank",
            "leave_style_top3_frequency",
        ]
    ].to_csv(output_dir / "q2_anchor_recovery_E111.csv", index=False)

    by_lens = (
        comparisons.groupby("lens", sort=False)
        .agg(
            mean_jaccard=("jaccard", "mean"),
            median_jaccard=("jaccard", "median"),
            exact_top3_targets=("exact_top3", "sum"),
        )
        .reset_index()
    )
    manifest = {
        "experiment": EXPERIMENT_ID,
        "status": "statistical_consensus_complete_chemical_grading_pending",
        "candidate_count": int(result["feature_code"].nunique()),
        "target_count": int(result["target_code"].nunique()),
        "top3_rows": len(top3),
        "lenses": [LENS_TOTAL, LENS_BOOTSTRAP, LENS_STYLE, LENS_DELETE],
        "top3_statistical_tier_counts": {
            str(key): int(value)
            for key, value in top3["statistical_robustness_tier"].value_counts().items()
        },
        "target_predictability_counts": {
            str(key): int(value)
            for key, value in top3.drop_duplicates("target_code")[
                "predictability_tier"
            ].value_counts().items()
        },
        "lens_agreement": by_lens.set_index("lens").to_dict(orient="index"),
        "direct_anchor_top3_count": int(
            anchor_audit["consensus_rank"].fillna(np.inf).le(3).sum()
        ),
        "E110_decision": "not_run: graph penalty did not improve OOF and route-B stability remained the priority",
        "gpu_used": False,
    }
    (output_dir / "q2_run_manifest_E111.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
