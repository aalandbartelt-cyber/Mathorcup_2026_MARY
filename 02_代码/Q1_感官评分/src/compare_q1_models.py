"""Paired beer-level bootstrap comparisons for the frozen Q1 OOF predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_contract import RESULT_DIR


RANDOM_STATE = 20261023
N_BOOTSTRAP = 20_000
COMPARATORS = [
    ("E001", "shrunk_style_mean"),
    ("E002", "pls2_quantile"),
    ("E002", "ridge_quantile"),
]


def main() -> None:
    extra = pd.read_csv(RESULT_DIR / "q1_oof_predictions_mean_E003.csv.gz")
    base = extra[["beer", "target_code", "y_true", "y_pred_mean"]].rename(
        columns={"y_pred_mean": "extra_trees"}
    )
    rng = np.random.default_rng(RANDOM_STATE)
    summary_rows = []
    target_rows = []
    extra_repeat = pd.read_csv(RESULT_DIR / "q1_model_comparison_E003.csv")

    for experiment_id, comparator in COMPARATORS:
        other_prediction = pd.read_csv(
            RESULT_DIR / f"q1_oof_predictions_mean_{experiment_id}.csv.gz"
        )
        other_prediction = other_prediction.loc[
            other_prediction["model"].eq(comparator),
            ["beer", "target_code", "y_pred_mean"],
        ].rename(columns={"y_pred_mean": comparator})
        paired = base.merge(
            other_prediction,
            on=["beer", "target_code"],
            how="inner",
            validate="one_to_one",
        )
        paired["extra_abs_error"] = (paired["y_true"] - paired["extra_trees"]).abs()
        paired["other_abs_error"] = (paired["y_true"] - paired[comparator]).abs()
        paired["delta_abs_error"] = (
            paired["extra_abs_error"] - paired["other_abs_error"]
        )
        per_beer = paired.groupby("beer", as_index=False).agg(
            delta_mae=("delta_abs_error", "mean")
        )
        bootstrap_index = rng.integers(
            0, len(per_beer), size=(N_BOOTSTRAP, len(per_beer))
        )
        bootstrap_delta = per_beer["delta_mae"].to_numpy()[bootstrap_index].mean(axis=1)

        other_repeat = pd.read_csv(
            RESULT_DIR / f"q1_model_comparison_{experiment_id}.csv"
        )
        other_repeat = other_repeat.loc[
            other_repeat["model"].eq(comparator), ["repeat", "macro_mae"]
        ].rename(columns={"macro_mae": "other_macro_mae"})
        repeat_paired = extra_repeat[["repeat", "macro_mae"]].rename(
            columns={"macro_mae": "extra_macro_mae"}
        ).merge(other_repeat, on="repeat", validate="one_to_one")

        extra_mae = float(paired["extra_abs_error"].mean())
        other_mae = float(paired["other_abs_error"].mean())
        lower, upper = np.quantile(bootstrap_delta, [0.025, 0.975])
        summary_rows.append(
            {
                "model": "extra_trees",
                "comparator": comparator,
                "mean_oof_mae_extra_trees": extra_mae,
                "mean_oof_mae_comparator": other_mae,
                "delta_mae_extra_minus_comparator": extra_mae - other_mae,
                "relative_mae_improvement": (other_mae - extra_mae) / other_mae,
                "beer_bootstrap_ci_lower": float(lower),
                "beer_bootstrap_ci_upper": float(upper),
                "beers_with_lower_mae": int((per_beer["delta_mae"] < 0).sum()),
                "n_beers": len(per_beer),
                "outer_repeats_with_lower_macro_mae": int(
                    (repeat_paired["extra_macro_mae"] < repeat_paired["other_macro_mae"]).sum()
                ),
                "n_outer_repeats": len(repeat_paired),
                "bootstrap_resamples": N_BOOTSTRAP,
                "random_state": RANDOM_STATE,
            }
        )
        target_summary = paired.groupby("target_code", as_index=False).agg(
            extra_trees_mae=("extra_abs_error", "mean"),
            comparator_mae=("other_abs_error", "mean"),
        )
        target_summary["delta_mae_extra_minus_comparator"] = (
            target_summary["extra_trees_mae"] - target_summary["comparator_mae"]
        )
        target_summary["comparator"] = comparator
        target_rows.append(target_summary)

    summary = pd.DataFrame(summary_rows)
    by_target = pd.concat(target_rows, ignore_index=True)
    summary.to_csv(RESULT_DIR / "q1_pairwise_bootstrap_E003.csv", index=False)
    by_target.to_csv(RESULT_DIR / "q1_pairwise_by_target_E003.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
