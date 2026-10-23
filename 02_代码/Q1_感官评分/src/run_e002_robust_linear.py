"""Run E002 rank-Gaussian Ridge and PLS2 on the frozen Q1 folds."""

from __future__ import annotations

import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn

from baseline_models import (
    fit_predict_pls_quantile,
    fit_predict_ridge_quantile,
    regression_metrics,
    select_parameter_by_mae,
)
from data_contract import ATTACHMENTS, RESULT_DIR, load_q1


EXPERIMENT_ID = "E002"
INNER_SPLITS = 3
RANDOM_STATE = 20261023
RIDGE_ALPHA_CANDIDATES = [0.1, 1.0, 10.0, 100.0, 1000.0]
PLS_COMPONENT_CANDIDATES = [2, 5, 10, 15, 20]
MODELS = ["ridge_quantile", "pls2_quantile"]


def main() -> None:
    started = time.time()
    x_frame, y_frame, style = load_q1()
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    splits = pd.read_csv(RESULT_DIR / "q1_cv_splits.csv")
    x = x_frame.to_numpy(float)
    y = y_frame.to_numpy(float)
    styles = style.reset_index(drop=True)
    repeats = sorted(splits["repeat"].unique().tolist())
    predictions = {
        model: np.full((len(repeats), len(x), y.shape[1]), np.nan, dtype=np.float64)
        for model in MODELS
    }
    fold_assignment = np.full((len(repeats), len(x)), -1, dtype=np.int8)
    parameter_rows = []

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

        def ridge_inner(inner_train, inner_validation, candidate):
            return fit_predict_ridge_quantile(
                x_train[inner_train],
                y_train[inner_train],
                x_train[inner_validation],
                alpha=float(candidate),
                random_state=split_seed,
            )

        selected_alpha, alpha_scores = select_parameter_by_mae(
            targets=y_train,
            candidates=RIDGE_ALPHA_CANDIDATES,
            predict_fold=ridge_inner,
            n_splits=INNER_SPLITS,
            random_state=split_seed,
        )
        predictions["ridge_quantile"][repeat_position, validation_index] = (
            fit_predict_ridge_quantile(
                x_train,
                y_train,
                x_validation,
                alpha=float(selected_alpha),
                random_state=split_seed,
            )
        )

        def pls_inner(inner_train, inner_validation, candidate):
            return fit_predict_pls_quantile(
                x_train[inner_train],
                y_train[inner_train],
                x_train[inner_validation],
                n_components=int(candidate),
                random_state=split_seed,
            )

        selected_components, component_scores = select_parameter_by_mae(
            targets=y_train,
            candidates=PLS_COMPONENT_CANDIDATES,
            predict_fold=pls_inner,
            n_splits=INNER_SPLITS,
            random_state=split_seed,
        )
        predictions["pls2_quantile"][repeat_position, validation_index] = (
            fit_predict_pls_quantile(
                x_train,
                y_train,
                x_validation,
                n_components=int(selected_components),
                random_state=split_seed,
            )
        )
        fold_assignment[repeat_position, validation_index] = int(fold)
        parameter_rows.extend(
            [
                {
                    "experiment_id": EXPERIMENT_ID,
                    "repeat": repeat,
                    "fold": fold,
                    "model": "ridge_quantile",
                    "parameter": "alpha",
                    "selected_value": selected_alpha,
                    "inner_scores_json": json.dumps(alpha_scores, sort_keys=True),
                },
                {
                    "experiment_id": EXPERIMENT_ID,
                    "repeat": repeat,
                    "fold": fold,
                    "model": "pls2_quantile",
                    "parameter": "n_components",
                    "selected_value": selected_components,
                    "inner_scores_json": json.dumps(component_scores, sort_keys=True),
                },
            ]
        )
        print(
            f"[{split_number:02d}/{len(split_groups)}] repeat={repeat} fold={fold} "
            f"alpha={selected_alpha:g} pls={selected_components}"
        )

    if np.any(fold_assignment < 0):
        raise RuntimeError("some samples have no outer-fold assignment")
    for model, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"{model} produced non-finite OOF predictions")

    metric_rows = []
    for model, values in predictions.items():
        for repeat_position, repeat in enumerate(repeats):
            for target_position, target in enumerate(y_frame.columns):
                metric_rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "model": model,
                        "repeat": int(repeat),
                        "target_code": target,
                        **regression_metrics(
                            y[:, target_position],
                            values[repeat_position, :, target_position],
                        ),
                    }
                )
        mean_prediction = values.mean(axis=0)
        for target_position, target in enumerate(y_frame.columns):
            metric_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "model": model,
                    "repeat": "mean_oof",
                    "target_code": target,
                    **regression_metrics(
                        y[:, target_position], mean_prediction[:, target_position]
                    ),
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
    for model, values in predictions.items():
        prediction_mean = values.mean(axis=0)
        prediction_sd = values.std(axis=0, ddof=1)
        for sample_position, beer in enumerate(chemistry["beer"]):
            for target_position, target in enumerate(y_frame.columns):
                mean_prediction_rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "model": model,
                        "beer": int(beer),
                        "tasting_category_fine": styles.iloc[sample_position],
                        "target_code": target,
                        "y_true": y[sample_position, target_position],
                        "y_pred_mean": prediction_mean[sample_position, target_position],
                        "y_pred_sd_across_repeats": prediction_sd[
                            sample_position, target_position
                        ],
                    }
                )

    np.savez_compressed(
        RESULT_DIR / "q1_oof_predictions_E002.npz",
        y_true=y,
        beer=chemistry["beer"].to_numpy(int),
        style=styles.to_numpy(str),
        target_code=np.asarray(y_frame.columns, dtype=str),
        repeat=np.asarray(repeats, dtype=int),
        fold_assignment=fold_assignment,
        **{f"pred_{model}": value for model, value in predictions.items()},
    )
    pd.DataFrame(mean_prediction_rows).to_csv(
        RESULT_DIR / "q1_oof_predictions_mean_E002.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics.to_csv(RESULT_DIR / "q1_metrics_by_target_E002.csv", index=False)
    comparison.to_csv(RESULT_DIR / "q1_model_comparison_E002.csv", index=False)
    summary.to_csv(RESULT_DIR / "q1_model_summary_E002.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(
        RESULT_DIR / "q1_selected_parameters_E002.csv", index=False
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "models": MODELS,
        "outer_cv": "same frozen 10x3 splits as E001",
        "inner_cv": "3-fold shuffled KFold",
        "selection_metric": "macro multi-output MAE",
        "preprocessing": "fold-local median + rank-Gaussian + missing indicator",
        "random_state": RANDOM_STATE,
        "elapsed_seconds": round(time.time() - started, 3),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    (RESULT_DIR / "q1_run_manifest_E002.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\nModel summary")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

