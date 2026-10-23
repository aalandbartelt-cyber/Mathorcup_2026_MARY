"""Build Q2 E106 three-method rank consensus and PLS sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from q2_contract import RESULT_DIR
from rank_consensus import reciprocal_rank_consensus, signed_direction_consensus


MAIN_METHODS = (
    "E101_SpearmanTop3",
    "E103_ElasticNet",
    "E105_ExtraTreesConditionalPermutation",
)
SENSITIVITY_METHODS = MAIN_METHODS + ("E102_PLS_VIP",)


def _load_rankings(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(output_dir / "q2_method_rankings_E101_E104.csv.gz")
    tree = pd.read_csv(output_dir / "q2_conditional_importance_E105.csv")
    tree = tree.assign(direction="importance_only")
    rankings = pd.concat(
        [
            base.loc[base["method"].isin(SENSITIVITY_METHODS)],
            tree,
        ],
        ignore_index=True,
        sort=False,
    )
    return rankings, base, tree


def _enrich_consensus(
    consensus: pd.DataFrame,
    rankings: pd.DataFrame,
    base: pd.DataFrame,
    tree: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    direction = (
        rankings.loc[
            rankings["method"].isin(("E101_SpearmanTop3", "E103_ElasticNet")),
            ["target_code", "feature_code", "method", "direction"],
        ]
        .groupby(["target_code", "feature_code"], sort=False)["direction"]
        .agg(lambda values: signed_direction_consensus(values.tolist()))
        .rename("direction_consensus")
        .reset_index()
    )
    spearman = base.loc[base["method"].eq("E101_SpearmanTop3"), [
        "target_code",
        "feature_code",
        "median_score",
        "rank_iqr",
    ]].rename(
        columns={"median_score": "spearman_median", "rank_iqr": "spearman_rank_iqr"}
    )
    elastic = base.loc[base["method"].eq("E103_ElasticNet"), [
        "target_code",
        "feature_code",
        "median_score",
        "selection_frequency",
        "rank_iqr",
    ]].rename(
        columns={
            "median_score": "elasticnet_median_coefficient",
            "selection_frequency": "elasticnet_nonzero_frequency",
            "rank_iqr": "elasticnet_rank_iqr",
        }
    )
    tree_evidence = tree[[
        "target_code",
        "feature_code",
        "median_score",
        "top3_frequency",
        "positive_fraction",
        "score_iqr",
    ]].rename(
        columns={
            "median_score": "tree_median_delta_mae",
            "top3_frequency": "tree_top3_frequency",
            "positive_fraction": "tree_positive_fraction",
            "score_iqr": "tree_delta_mae_iqr",
        }
    )
    candidates = pd.read_csv(output_dir / "q2_candidate_dictionary.csv")
    targets = pd.read_csv(output_dir / "q2_target_dictionary.csv")
    enriched = (
        consensus.merge(direction, on=["target_code", "feature_code"], how="left")
        .merge(spearman, on=["target_code", "feature_code"], how="left")
        .merge(elastic, on=["target_code", "feature_code"], how="left")
        .merge(tree_evidence, on=["target_code", "feature_code"], how="left")
        .merge(
            candidates[
                [
                    "feature_code",
                    "compound_name",
                    "chemical_name_short",
                    "unit",
                    "factor_type",
                ]
            ],
            on="feature_code",
            how="left",
            validate="many_to_one",
        )
        .merge(
            targets[["target_code", "target_name_en", "target_group", "direct_anchor"]],
            on="target_code",
            how="left",
            validate="many_to_one",
        )
    )
    enriched["statistical_consensus_tier"] = np.select(
        [
            enriched["method_top10_support"].eq(3)
            & enriched["method_top3_support"].ge(2),
            enriched["method_top10_support"].ge(2)
            & enriched["method_top3_support"].ge(1),
        ],
        ["S1_strong", "S2_moderate"],
        default="S3_exploratory",
    )
    return enriched


def main(output_dir: Path = RESULT_DIR) -> None:
    output_dir = Path(output_dir)
    rankings, base, tree = _load_rankings(output_dir)
    main_input = rankings.loc[rankings["method"].isin(MAIN_METHODS)]
    sensitivity_input = rankings.loc[rankings["method"].isin(SENSITIVITY_METHODS)]
    main_consensus = reciprocal_rank_consensus(main_input, expected_methods=3, rrf_k=60)
    sensitivity = reciprocal_rank_consensus(
        sensitivity_input, expected_methods=4, rrf_k=60
    )
    enriched = _enrich_consensus(main_consensus, rankings, base, tree, output_dir)
    top3 = enriched.loc[enriched["consensus_rank"].le(3)].copy()
    sensitivity_top3 = sensitivity.loc[sensitivity["consensus_rank"].le(3)].copy()

    sensitivity_rows = []
    for target in sorted(top3["target_code"].unique()):
        main_set = set(top3.loc[top3["target_code"].eq(target), "feature_code"])
        alternate_set = set(
            sensitivity_top3.loc[
                sensitivity_top3["target_code"].eq(target), "feature_code"
            ]
        )
        sensitivity_rows.append(
            {
                "target_code": target,
                "intersection": len(main_set & alternate_set),
                "union": len(main_set | alternate_set),
                "jaccard": len(main_set & alternate_set) / len(main_set | alternate_set),
                "exact_top3": main_set == alternate_set,
            }
        )
    sensitivity_audit = pd.DataFrame(sensitivity_rows)

    targets = pd.read_csv(output_dir / "q2_target_dictionary.csv")
    anchor_audit = targets.loc[targets["direct_anchor"].notna()].merge(
        enriched,
        left_on=["target_code", "direct_anchor"],
        right_on=["target_code", "feature_code"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_consensus"),
    )
    anchor_audit = anchor_audit[
        [
            "target_code",
            "target_name_en",
            "direct_anchor",
            "consensus_rank",
            "method_top3_support",
            "method_top10_support",
            "rrf_score",
            "direction_consensus",
            "statistical_consensus_tier",
        ]
    ].sort_values("target_code")

    enriched.to_csv(output_dir / "q2_consensus_rankings_E106.csv", index=False)
    top3.to_csv(output_dir / "q2_top3_by_target_E106.csv", index=False)
    sensitivity.to_csv(
        output_dir / "q2_consensus_rankings_E106_four_method_sensitivity.csv",
        index=False,
    )
    sensitivity_audit.to_csv(
        output_dir / "q2_e106_four_method_sensitivity.csv", index=False
    )
    anchor_audit.to_csv(output_dir / "q2_anchor_recovery_E106.csv", index=False)
    report = {
        "experiment": "E106",
        "status": "complete",
        "main_methods": list(MAIN_METHODS),
        "aggregation": "equal-weight support-aware reciprocal rank fusion",
        "rrf_k": 60,
        "targets": int(top3["target_code"].nunique()),
        "top3_rows": len(top3),
        "consensus_tier_counts": {
            str(key): int(value)
            for key, value in top3["statistical_consensus_tier"]
            .value_counts()
            .items()
        },
        "direction_counts": {
            str(key): int(value)
            for key, value in top3["direction_consensus"].value_counts().items()
        },
        "four_method_sensitivity": {
            "mean_jaccard": float(sensitivity_audit["jaccard"].mean()),
            "median_jaccard": float(sensitivity_audit["jaccard"].median()),
            "exact_top3_targets": int(sensitivity_audit["exact_top3"].sum()),
        },
        "gpu_used": False,
        "decision": "E106 is a candidate consensus; final Top-3 requires E107/E108",
    }
    (output_dir / "q2_run_manifest_E106.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
