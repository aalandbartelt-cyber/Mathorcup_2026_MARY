"""Create machine-readable E105 performance, agreement, and anchor-module audits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from q2_contract import RESULT_DIR


METHOD_E105 = "E105_ExtraTreesConditionalPermutation"


def main(output_dir: Path = RESULT_DIR) -> None:
    output_dir = Path(output_dir)
    base_repeat = pd.read_csv(
        output_dir / "q2_model_summary_by_repeat_E101_E104.csv"
    )
    tree_repeat = pd.read_csv(output_dir / "q2_model_summary_by_repeat_E105.csv")
    comparisons = []
    for comparator in (
        "E101_SpearmanTop3",
        "E102_PLS_VIP",
        "E103_ElasticNet",
        "E104_GraphElasticNet",
    ):
        left = tree_repeat.set_index("repeat")
        right = base_repeat.loc[base_repeat["method"].eq(comparator)].set_index("repeat")
        for repeat in sorted(left.index):
            comparisons.append(
                {
                    "repeat": int(repeat),
                    "method": METHOD_E105,
                    "comparator": comparator,
                    "delta_mae_E105_minus_comparator": left.loc[repeat, "macro_mae"]
                    - right.loc[repeat, "macro_mae"],
                    "delta_r2_E105_minus_comparator": left.loc[repeat, "macro_r2"]
                    - right.loc[repeat, "macro_r2"],
                    "delta_spearman_E105_minus_comparator": left.loc[
                        repeat, "macro_spearman"
                    ]
                    - right.loc[repeat, "macro_spearman"],
                }
            )
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output_dir / "q2_e105_paired_performance.csv", index=False)

    tree_top3 = pd.read_csv(output_dir / "q2_top3_E105.csv")
    base_top3 = pd.read_csv(output_dir / "q2_top3_by_method_E101_E104.csv")
    agreement_rows = []
    population = 218
    selected_count = 3
    expected = selected_count**2 / population
    for comparator in sorted(base_top3["method"].unique()):
        for target in sorted(tree_top3["target_code"].unique()):
            left = set(
                tree_top3.loc[tree_top3["target_code"].eq(target), "feature_code"]
            )
            right = set(
                base_top3.loc[
                    base_top3["method"].eq(comparator)
                    & base_top3["target_code"].eq(target),
                    "feature_code",
                ]
            )
            intersection = len(left & right)
            union = len(left | right)
            agreement_rows.append(
                {
                    "target_code": target,
                    "method_a": METHOD_E105,
                    "method_b": comparator,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union,
                    "kuncheva": (intersection - expected) / (selected_count - expected),
                }
            )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(output_dir / "q2_method_agreement_E105.csv", index=False)

    modules = pd.read_csv(output_dir / "q2_module_importance_folds_E105.csv.gz")
    modules["module_rank"] = modules.groupby(
        ["split_id", "target_code"]
    )["delta_mae"].rank(method="first", ascending=False)
    membership = (
        modules[["split_id", "module_id", "module_size", "module_members"]]
        .drop_duplicates()
        .assign(feature_code=lambda frame: frame["module_members"].str.split("|"))
        .explode("feature_code")
    )
    targets = pd.read_csv(output_dir / "q2_target_dictionary.csv")
    anchors = targets.loc[targets["direct_anchor"].notna(), [
        "target_code",
        "target_name_en",
        "direct_anchor",
    ]]
    anchor_modules = anchors.merge(
        membership,
        left_on="direct_anchor",
        right_on="feature_code",
        how="left",
    ).merge(
        modules[
            [
                "split_id",
                "module_id",
                "target_code",
                "delta_mae",
                "module_rank",
            ]
        ],
        on=["split_id", "module_id", "target_code"],
        how="left",
    )
    anchor_summary = (
        anchor_modules.groupby(
            ["target_code", "target_name_en", "direct_anchor"], dropna=False
        )
        .agg(
            folds_with_eligible_anchor=("delta_mae", "count"),
            median_module_size=("module_size", "median"),
            median_module_rank=("module_rank", "median"),
            module_top3_frequency=("module_rank", lambda x: float((x <= 3).mean())),
            median_module_delta_mae=("delta_mae", "median"),
            positive_module_fraction=("delta_mae", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    anchor_summary.to_csv(output_dir / "q2_anchor_module_recovery_E105.csv", index=False)

    e103 = comparison.loc[comparison["comparator"].eq("E103_ElasticNet")]
    report = {
        "experiment": "E105",
        "performance_vs_E103": {
            "mean_delta_mae": float(e103["delta_mae_E105_minus_comparator"].mean()),
            "mean_delta_r2": float(e103["delta_r2_E105_minus_comparator"].mean()),
            "mean_delta_spearman": float(
                e103["delta_spearman_E105_minus_comparator"].mean()
            ),
            "mae_wins": int((e103["delta_mae_E105_minus_comparator"] < 0).sum()),
            "n_repeats": len(e103),
        },
        "top3_stability": {
            "median_top3_frequency": float(tree_top3["top3_frequency"].median()),
            "minimum_top3_frequency": float(tree_top3["top3_frequency"].min()),
            "nonpositive_median_importance_in_top3": int(
                (tree_top3["median_score"] <= 0).sum()
            ),
        },
        "agreement": {
            method: {
                "mean_jaccard": float(frame["jaccard"].mean()),
                "median_jaccard": float(frame["jaccard"].median()),
                "mean_kuncheva": float(frame["kuncheva"].mean()),
            }
            for method, frame in agreement.groupby("method_b")
        },
        "decision": "promote E105 prediction model; do not finalize compounds before E106-E108",
    }
    (output_dir / "q2_e105_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
