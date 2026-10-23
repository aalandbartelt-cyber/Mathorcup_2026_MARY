"""Run E003 native multi-output ExtraTrees on the frozen Q1 folds."""

from __future__ import annotations

import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn

from baseline_models import (
    fit_predict_extra_trees,
    regression_metrics,
    select_parameter_by_mae,
)
from data_contract import ATTACHMENTS, RESULT_DIR, load_q1


EXPERIMENT_ID = "E003"
MODEL = "extra_trees"
INNER_SPLITS = 3
RANDOM_STATE = 20261023
N_ESTIMATORS = 300
CANDIDATES = [
    f"max_features={max_features:.2f},min_samples_leaf={min_samples_leaf}"
    for max_features in (0.33, 0.60, 1.00)
    for min_samples_leaf in (1, 2, 4, 8)
]


def parse_candidate(candidate: str) -> tuple[float, int]:
    values = dict(item.split("=") for item in candidate.split(","))
    return float(values["max_features"]), int(values["min_samples_leaf"])


def main() -> None:
    started = time.time()
    x_frame, y_frame, style = load_q1()
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    splits = pd.read_csv(RESULT_DIR / "q1_cv_splits.csv")
    x = x_frame.to_numpy(float)
    y = y_frame.to_numpy(float)
    styles = style.reset_index(drop=True)
    repeats = sorted(splits["repeat"].unique().tolist())
    predictions = np.full(
        (len(repeats), len(x), y.shape[1]), np.nan, dtype=np.float64
    )
    fold_assignment = np.full((len(repeats), len(x)), -1, dtype=np.int8)
    parameter_rows: list[dict] = []

    split_groups = list(splits.groupby(["repeat", "fold"], sort=True))
    for split_number, ((repeat, fold), split_frame) in enumerate(split_groups, start=1):
        validation_index = split_frame["row_position"].to_numpy(int)
        validation_mask = np.zeros(len(x), dtype=bool)
        validation_mask[validation_index] = True
        train_index = np.flatnonzero(~validation_mask)
        repeat_position = repeats.index(repeat)
        x_train, x_validation = x[train_index], x[validation_index]
        y_train = y[train_index]
        split_seed = RANDOM_STATE + int(repeat) * 10 + int(fold)

        def tree_inner(inner_train, inner_validation, candidate):
            max_features, min_samples_leaf = parse_candidate(candidate)
            return fit_predict_extra_trees(
                x_train[inner_train],
                y_train[inner_train],
                x_train[inner_validation],
                n_estimators=N_ESTIMATORS,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                max_depth=None,
                random_state=split_seed,
            )

        selected, scores = select_parameter_by_mae(
            targets=y_train,
            candidates=CANDIDATES,
            predict_fold=tree_inner,
            n_splits=INNER_SPLITS,
            random_state=split_seed,
        )
        max_features, min_samples_leaf = parse_candidate(selected)
        predictions[repeat_position, validation_index] = fit_predict_extra_trees(
            x_train,
            y_train,
            x_validation,
            n_estimators=N_ESTIMATORS,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_depth=None,
            random_state=split_seed,
        )
        fold_assignment[repeat_position, validation_index] = int(fold)
        parameter_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "repeat": repeat,
                "fold": fold,
                "model": MODEL,
                "selected_candidate": selected,
                "selected_max_features": max_features,
                "selected_min_samples_leaf": min_samples_leaf,
                "inner_scores_json": json.dumps(scores, sort_keys=True),
            }
        )
        print(
            f"[{split_number:02d}/{len(split_groups)}] repeat={repeat} fold={fold} "
            f"max_features={max_features:.2f} leaf={min_samples_leaf}",
            flush=True,
        )

    if np.any(fold_assignment < 0) or not np.isfinite(predictions).all():
        raise RuntimeError("incomplete or non-finite E003 OOF predictions")

    metric_rows = []
    for repeat_position, repeat in enumerate(repeats):
        for target_position, target in enumerate(y_frame.columns):
            metric_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "model": MODEL,
                    "repeat": int(repeat),
                    "target_code": target,
                    **regression_metrics(
                        y[:, target_position],
                        predictions[repeat_position, :, target_position],
                    ),
                }
            )
    mean_prediction = predictions.mean(axis=0)
    prediction_sd = predictions.std(axis=0, ddof=1)
    for target_position, target in enumerate(y_frame.columns):
        metric_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "model": MODEL,
                "repeat": "mean_oof",
                "target_code": target,
                **regression_metrics(y[:, target_position], mean_prediction[:, target_position]),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    repeat_metrics = metrics.loc[metrics["repeat"] != "mean_oof"].copy()
    numeric_metrics = [
        "r2",
        "mae",
        "rmse",
        "spearman",
        "calibration_intercept",
        "calibration_slope",
    ]
    comparison = (
        repeat_metrics.groupby(["model", "repeat"], as_index=False)[numeric_metrics]
        .mean(numeric_only=True)
        .rename(columns={column: f"macro_{column}" for column in numeric_metrics})
    )
    summary = comparison.groupby("model").agg(
        repeats=("repeat", "count"),
        macro_r2_mean=("macro_r2", "mean"),
        macro_r2_sd=("macro_r2", "std"),
        macro_mae_mean=("macro_mae", "mean"),
        macro_mae_sd=("macro_mae", "std"),
        macro_rmse_mean=("macro_rmse", "mean"),
        macro_spearman_mean=("macro_spearman", "mean"),
    ).reset_index()

    mean_prediction_rows = []
    for sample_position, beer in enumerate(chemistry["beer"]):
        for target_position, target in enumerate(y_frame.columns):
            mean_prediction_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "model": MODEL,
                    "beer": int(beer),
                    "tasting_category_fine": styles.iloc[sample_position],
                    "target_code": target,
                    "y_true": y[sample_position, target_position],
                    "y_pred_mean": mean_prediction[sample_position, target_position],
                    "y_pred_sd_across_repeats": prediction_sd[
                        sample_position, target_position
                    ],
                }
            )

    style_rows = []
    for style_name in sorted(styles.unique()):
        mask = styles.eq(style_name).to_numpy()
        residual = y[mask] - mean_prediction[mask]
        style_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "model": MODEL,
                "tasting_category_fine": style_name,
                "n_beers": int(mask.sum()),
                "mae_all_targets": float(np.mean(np.abs(residual))),
                "rmse_all_targets": float(np.mean(residual**2) ** 0.5),
                "mean_residual": float(np.mean(residual)),
            }
        )

    np.savez_compressed(
        RESULT_DIR / "q1_oof_predictions_E003.npz",
        y_true=y,
        beer=chemistry["beer"].to_numpy(int),
        style=styles.to_numpy(str),
        target_code=np.asarray(y_frame.columns, dtype=str),
        repeat=np.asarray(repeats, dtype=int),
        fold_assignment=fold_assignment,
        pred_extra_trees=predictions,
    )
    pd.DataFrame(mean_prediction_rows).to_csv(
        RESULT_DIR / "q1_oof_predictions_mean_E003.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics.to_csv(RESULT_DIR / "q1_metrics_by_target_E003.csv", index=False)
    comparison.to_csv(RESULT_DIR / "q1_model_comparison_E003.csv", index=False)
    summary.to_csv(RESULT_DIR / "q1_model_summary_E003.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(
        RESULT_DIR / "q1_selected_parameters_E003.csv", index=False
    )
    pd.DataFrame(style_rows).to_csv(
        RESULT_DIR / "q1_metrics_by_style_E003.csv", index=False
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "model": MODEL,
        "n_estimators": N_ESTIMATORS,
        "candidate_count": len(CANDIDATES),
        "outer_cv": "same frozen 10x3 splits as E001/E002",
        "inner_cv": "3-fold shuffled KFold",
        "selection_metric": "macro multi-output MAE",
        "preprocessing": "fold-local median + missing indicator",
        "random_state": RANDOM_STATE,
        "elapsed_seconds": round(time.time() - started, 3),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    (RESULT_DIR / "q1_run_manifest_E003.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\nModel summary")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
