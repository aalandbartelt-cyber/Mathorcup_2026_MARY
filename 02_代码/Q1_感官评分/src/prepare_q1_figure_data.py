"""Prepare leakage-safe, frozen data tables for the seven Q1 draft figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_models import empirical_cdf_grade
from data_contract import ATTACHMENTS, RESULT_DIR, load_q1, target_dictionary


FIGURE_DIR = RESULT_DIR / "figure_data"
REPRESENTATIVE_TARGET_COUNT = 2


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _, y_frame, style = load_q1()
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    dictionary = target_dictionary(y_frame)
    dictionary.to_csv(FIGURE_DIR / "target_dictionary.csv", index=False)

    distribution = y_frame.copy()
    distribution.insert(0, "beer", chemistry["beer"].to_numpy())
    distribution = distribution.melt(
        id_vars="beer", var_name="target_code", value_name="z_score"
    ).merge(dictionary, on="target_code", how="left", validate="many_to_one")
    distribution.to_csv(
        FIGURE_DIR / "target_distribution.csv.gz", index=False, compression="gzip"
    )

    model_specs = [
        ("E001", "global_mean", "全局均值"),
        ("E001", "shrunk_style_mean", "风格收缩均值"),
        ("E002", "pls2_quantile", "PLS2（秩高斯）"),
        ("E002", "ridge_quantile", "Ridge（秩高斯）"),
        ("E003", "extra_trees", "ExtraTrees"),
    ]
    comparison_parts = []
    for experiment_id, model, display_name in model_specs:
        frame = pd.read_csv(RESULT_DIR / f"q1_model_comparison_{experiment_id}.csv")
        frame = frame.loc[frame["model"].eq(model)].copy()
        frame["display_name"] = display_name
        comparison_parts.append(frame)
    comparison = pd.concat(comparison_parts, ignore_index=True)
    comparison.to_csv(FIGURE_DIR / "model_comparison.csv", index=False)

    metrics = pd.read_csv(RESULT_DIR / "q1_metrics_by_target_E003.csv")
    metrics = metrics.loc[metrics["repeat"].astype(str).eq("mean_oof")].copy()
    metrics = metrics.merge(dictionary, on="target_code", how="left", validate="one_to_one")
    metrics.to_csv(FIGURE_DIR / "target_metrics_extra_trees.csv", index=False)

    ordered = metrics.sort_values("r2").reset_index(drop=True)
    chosen = pd.concat(
        [
            ordered.head(REPRESENTATIVE_TARGET_COUNT).assign(tier="低"),
            ordered.iloc[
                len(ordered) // 2 - 1 : len(ordered) // 2 + 1
            ].assign(tier="中"),
            ordered.tail(REPRESENTATIVE_TARGET_COUNT).assign(tier="高"),
        ],
        ignore_index=True,
    )
    chosen["facet_label"] = chosen.apply(
        lambda row: (
            f"{row['tier']}：{row['target_code']}  "
            f"R²={row['r2']:.2f}, MAE={row['mae']:.2f}"
        ),
        axis=1,
    )

    prediction = pd.read_csv(
        RESULT_DIR / "q1_oof_predictions_mean_E003.csv.gz"
    ).merge(
        chosen[["target_code", "tier", "facet_label"]],
        on="target_code",
        how="inner",
        validate="many_to_one",
    )
    prediction["residual"] = prediction["y_true"] - prediction["y_pred_mean"]
    prediction.to_csv(
        FIGURE_DIR / "representative_oof.csv.gz", index=False, compression="gzip"
    )

    style_metrics = pd.read_csv(RESULT_DIR / "q1_metrics_by_style_E003.csv")
    style_metrics.to_csv(FIGURE_DIR / "style_metrics_extra_trees.csv", index=False)

    archive = np.load(RESULT_DIR / "q1_oof_predictions_E003.npz")
    y = archive["y_true"]
    predictions = archive["pred_extra_trees"]
    fold_assignment = archive["fold_assignment"]
    grades = np.full_like(predictions, np.nan, dtype=float)
    for repeat_position in range(predictions.shape[0]):
        for fold in np.unique(fold_assignment[repeat_position]):
            validation_mask = fold_assignment[repeat_position] == fold
            training_mask = ~validation_mask
            grades[repeat_position, validation_mask] = empirical_cdf_grade(
                y[training_mask], predictions[repeat_position, validation_mask]
            )
    if not np.isfinite(grades).all() or grades.min() < 0 or grades.max() > 6:
        raise RuntimeError("invalid empirical-CDF OOF grades")
    mean_grade = grades.mean(axis=0)
    grade_rows = []
    for sample_position, beer in enumerate(chemistry["beer"]):
        for target_position, target in enumerate(y_frame.columns):
            grade_rows.append(
                {
                    "beer": int(beer),
                    "tasting_category_fine": style.iloc[sample_position],
                    "target_code": target,
                    "y_true_z": y[sample_position, target_position],
                    "y_pred_z": predictions[:, sample_position, target_position].mean(),
                    "grade_0_6": mean_grade[sample_position, target_position],
                    "grade_sd_across_repeats": grades[
                        :, sample_position, target_position
                    ].std(ddof=1),
                }
            )
    pd.DataFrame(grade_rows).to_csv(
        RESULT_DIR / "q1_oof_grade_E003.csv.gz", index=False, compression="gzip"
    )

    example_targets = ["acidity", "alcohol", "A_malt_all", "F_esters_flower"]
    repeat_position = 0
    fold = 0
    training_mask = fold_assignment[repeat_position] != fold
    mapping_rows = []
    for target in example_targets:
        target_position = y_frame.columns.get_loc(target)
        reference = y[training_mask, target_position]
        grid = np.linspace(reference.min() - 0.1, reference.max() + 0.1, 240)
        grade = empirical_cdf_grade(
            reference[:, None], grid[:, None]
        )[:, 0]
        mapping_rows.extend(
            {
                "target_code": target,
                "z_score": float(z_value),
                "grade_0_6": float(grade_value),
                "reference_repeat": 0,
                "reference_fold": 0,
                "n_training": int(training_mask.sum()),
            }
            for z_value, grade_value in zip(grid, grade)
        )
    pd.DataFrame(mapping_rows).to_csv(FIGURE_DIR / "scale_mapping_example.csv", index=False)

    print(
        {
            "figure_data_dir": str(FIGURE_DIR),
            "representative_targets": chosen["target_code"].tolist(),
            "grade_range": [float(grades.min()), float(grades.max())],
            "model_rows": len(comparison),
        }
    )


if __name__ == "__main__":
    main()
