"""Run E005 leakage-safe continuous ensemble on frozen Q1 folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from baseline_models import (
    fit_predict_extra_trees,
    fit_predict_pls_quantile,
    fit_predict_ridge_quantile,
)
from data_contract import ATTACHMENTS, RESULT_DIR, load_q1
from ensemble_models import (
    apply_convex_blend,
    fit_predict_hgb_multioutput,
    select_feasible_convex_blend,
    simplex_weight_grid,
)
from q1_metrics import matrix_metrics
from second_round_runner import (
    load_config,
    load_run_checkpoint,
    make_inner_splits,
    outer_indices,
    repeats_for_stage,
    save_prediction_archive,
    save_run_checkpoint,
    write_metric_artifacts,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "E005_continuous_ensemble.yaml"


def _candidates(config: dict, model: str) -> list[object]:
    if model == "extra_trees":
        return [
            {"max_features": float(max_features), "min_samples_leaf": int(leaf)}
            for max_features in config["extra_trees"]["max_features"]
            for leaf in config["extra_trees"]["min_samples_leaf"]
        ]
    if model == "hgb":
        return list(config["hgb_profiles"])
    if model == "ridge_quantile":
        return [float(value) for value in config["ridge_alphas"]]
    if model == "pls2_quantile":
        return [int(value) for value in config["pls_components"]]
    raise ValueError(f"unsupported E005 base model: {model}")


def _fit_base(
    model: str,
    parameter: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    config: dict,
    random_state: int,
) -> np.ndarray:
    if model == "extra_trees":
        values = dict(parameter)
        return fit_predict_extra_trees(
            x_train,
            y_train,
            x_validation,
            n_estimators=int(config["extra_trees"]["n_estimators"]),
            max_features=float(values["max_features"]),
            min_samples_leaf=int(values["min_samples_leaf"]),
            max_depth=config["extra_trees"]["max_depth"],
            random_state=random_state,
            n_jobs=1,
        )
    if model == "hgb":
        values = dict(parameter)
        return fit_predict_hgb_multioutput(
            x_train,
            y_train,
            x_validation,
            max_leaf_nodes=int(values["max_leaf_nodes"]),
            min_samples_leaf=int(values["min_samples_leaf"]),
            learning_rate=float(values["learning_rate"]),
            l2_regularization=float(values["l2_regularization"]),
            max_iter=int(values["max_iter"]),
            random_state=random_state,
        )
    if model == "ridge_quantile":
        return fit_predict_ridge_quantile(
            x_train,
            y_train,
            x_validation,
            alpha=float(parameter),
            random_state=random_state,
        )
    if model == "pls2_quantile":
        return fit_predict_pls_quantile(
            x_train,
            y_train,
            x_validation,
            n_components=int(parameter),
            random_state=random_state,
        )
    raise ValueError(f"unsupported E005 base model: {model}")


def _parameter_key(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        metrics["mae"],
        -metrics["r2"],
        -metrics["spearman"],
        -metrics["qwk_proxy"],
        -metrics["acc_pm1_proxy"],
    )


def run(stage: str) -> dict:
    started = time.time()
    config = load_config(CONFIG_PATH)
    experiment_id = str(config["experiment_id"])
    repeats = repeats_for_stage(config, stage)
    suffix = "screening" if stage == "screening" else ""
    base_models = list(config["base_models"])
    output_models = [*base_models, "continuous_blend"]

    x_frame, y_frame, style = load_q1()
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    splits_path = RESULT_DIR / "q1_cv_splits.csv"
    splits = pd.read_csv(splits_path)
    x = x_frame.to_numpy(float)
    y = y_frame.to_numpy(float)
    predictions = np.full(
        (len(repeats), len(y), y.shape[1], len(output_models)),
        np.nan,
        dtype=float,
    )
    fold_assignment = np.full((len(repeats), len(y)), -1, dtype=np.int8)
    parameter_rows: list[dict] = []

    config_hash = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    signature_payload = {
        "experiment_id": experiment_id,
        "stage": stage,
        "repeats": repeats,
        "config_sha256": config_hash,
        "cv_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "chemistry_sha256": hashlib.sha256(
            (ATTACHMENTS / "1.csv").read_bytes()
        ).hexdigest(),
        "target_sha256": hashlib.sha256(
            (ATTACHMENTS / "4.csv").read_bytes()
        ).hexdigest(),
        "shape": list(y.shape),
        "output_models": output_models,
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint_path = RESULT_DIR / (
        f"q1_checkpoint_{experiment_id}"
        + (f"_{suffix}" if suffix else "")
        + ".npz"
    )
    if checkpoint_path.exists():
        restored = load_run_checkpoint(
            checkpoint_path,
            run_signature=run_signature,
            prediction_shape=predictions.shape,
            assignment_shape=fold_assignment.shape,
        )
        predictions = restored["predictions"]
        fold_assignment = restored["fold_assignment"]
        parameter_rows = restored["parameter_rows"]
        print(f"[resume] loaded {checkpoint_path}", flush=True)

    selected_split_groups = [
        ((int(repeat), int(fold)), frame)
        for (repeat, fold), frame in splits.groupby(["repeat", "fold"], sort=True)
        if int(repeat) in repeats
    ]
    weight_candidates = simplex_weight_grid(
        n_models=len(base_models), step=float(config["weight_step"])
    )

    for split_number, ((repeat, fold), _) in enumerate(selected_split_groups, start=1):
        train_index, validation_index = outer_indices(
            splits, repeat=repeat, fold=fold, n_samples=len(x)
        )
        repeat_position = repeats.index(repeat)
        already_complete = (
            np.isfinite(predictions[repeat_position, validation_index]).all()
            and np.all(fold_assignment[repeat_position, validation_index] == fold)
            and any(
                int(row["repeat"]) == repeat and int(row["fold"]) == fold
                for row in parameter_rows
            )
        )
        if already_complete:
            print(
                f"[{split_number:02d}/{len(selected_split_groups)}] "
                f"repeat={repeat} fold={fold} resumed",
                flush=True,
            )
            continue

        x_train, x_validation = x[train_index], x[validation_index]
        y_train = y[train_index]
        split_seed = int(config["random_state"]) + repeat * 10 + fold
        inner_splits = make_inner_splits(
            len(train_index), int(config["inner_splits"]), split_seed
        )
        selected_parameters: dict[str, object] = {}
        selected_inner_predictions: dict[str, np.ndarray] = {}
        candidate_metric_log: dict[str, list[dict]] = {}

        for model_position, model in enumerate(base_models):
            candidate_rows = []
            for candidate_position, candidate in enumerate(_candidates(config, model)):
                inner_prediction = np.full_like(y_train, np.nan, dtype=float)
                for inner_fold, (inner_train, inner_validation) in enumerate(
                    inner_splits
                ):
                    inner_prediction[inner_validation] = _fit_base(
                        model,
                        candidate,
                        x_train[inner_train],
                        y_train[inner_train],
                        x_train[inner_validation],
                        config,
                        random_state=(
                            split_seed
                            + model_position * 100_000
                            + candidate_position * 1_000
                            + inner_fold
                        ),
                    )
                if not np.isfinite(inner_prediction).all():
                    raise RuntimeError(f"E005 {model} inner OOF is incomplete")
                candidate_rows.append(
                    {
                        "parameter": candidate,
                        "metrics": matrix_metrics(y_train, inner_prediction),
                        "prediction": inner_prediction,
                    }
                )
            selected = min(candidate_rows, key=lambda row: _parameter_key(row["metrics"]))
            selected_parameters[model] = selected["parameter"]
            selected_inner_predictions[model] = selected["prediction"]
            candidate_metric_log[model] = [
                {"parameter": row["parameter"], "metrics": row["metrics"]}
                for row in candidate_rows
            ]

        inner_stack = np.stack(
            [selected_inner_predictions[model] for model in base_models], axis=-1
        )
        base_inner_metrics = {
            model: matrix_metrics(y_train, selected_inner_predictions[model])
            for model in base_models
        }
        tolerance = config["continuous_tolerance"]
        r2_floor = max(row["r2"] for row in base_inner_metrics.values()) - float(
            tolerance["r2_drop"]
        )
        mae_ceiling = min(row["mae"] for row in base_inner_metrics.values()) + float(
            tolerance["mae_increase"]
        )
        blend = select_feasible_convex_blend(
            y_train,
            inner_stack,
            weight_candidates,
            r2_floor=r2_floor,
            mae_ceiling=mae_ceiling,
        )

        outer_base_predictions = []
        for model_position, model in enumerate(base_models):
            outer_base_predictions.append(
                _fit_base(
                    model,
                    selected_parameters[model],
                    x_train,
                    y_train,
                    x_validation,
                    config,
                    random_state=split_seed + model_position * 100_000,
                )
            )
        outer_stack = np.stack(outer_base_predictions, axis=-1)
        outer_blend = apply_convex_blend(outer_stack, blend.weights)
        predictions[repeat_position, validation_index, :, : len(base_models)] = (
            outer_stack
        )
        predictions[repeat_position, validation_index, :, -1] = outer_blend
        fold_assignment[repeat_position, validation_index] = fold
        parameter_rows.append(
            {
                "experiment_id": experiment_id,
                "stage": stage,
                "repeat": repeat,
                "fold": fold,
                "selected_parameters_json": json.dumps(
                    selected_parameters, ensure_ascii=False, sort_keys=True
                ),
                "candidate_metrics_json": json.dumps(
                    candidate_metric_log, ensure_ascii=False, sort_keys=True
                ),
                "base_inner_metrics_json": json.dumps(
                    base_inner_metrics, sort_keys=True
                ),
                "blend_weights_json": json.dumps(blend.weights.tolist()),
                "blend_metrics_json": json.dumps(blend.metrics, sort_keys=True),
                "blend_feasible": blend.feasible,
                "r2_floor": r2_floor,
                "mae_ceiling": mae_ceiling,
            }
        )
        save_run_checkpoint(
            checkpoint_path,
            run_signature=run_signature,
            predictions=predictions,
            fold_assignment=fold_assignment,
            parameter_rows=parameter_rows,
        )
        print(
            f"[{split_number:02d}/{len(selected_split_groups)}] "
            f"repeat={repeat} fold={fold} weights="
            f"{np.round(blend.weights, 2).tolist()} feasible={blend.feasible}",
            flush=True,
        )

    prediction_map = {
        f"pred_{model}": predictions[:, :, :, position]
        for position, model in enumerate(output_models)
    }
    archive_name = f"q1_oof_predictions_{experiment_id}"
    if suffix:
        archive_name += f"_{suffix}"
    archive_path = RESULT_DIR / f"{archive_name}.npz"
    save_prediction_archive(
        archive_path,
        y_true=y,
        beer=chemistry["beer"].to_numpy(int),
        style=style.to_numpy(str),
        target_code=np.asarray(y_frame.columns, dtype=str),
        repeats=np.asarray(repeats, dtype=int),
        fold_assignment=fold_assignment,
        predictions=prediction_map,
    )
    summary_path, by_target_path = write_metric_artifacts(
        experiment_id,
        archive_path,
        prediction_keys=list(prediction_map),
        suffix=suffix,
    )
    parameter_path = RESULT_DIR / (
        f"q1_selected_parameters_{experiment_id}"
        + (f"_{suffix}" if suffix else "")
        + ".csv"
    )
    pd.DataFrame(parameter_rows).to_csv(parameter_path, index=False)
    manifest = {
        "experiment_id": experiment_id,
        "stage": stage,
        "status": "complete",
        "repeats": repeats,
        "outer_folds": len(selected_split_groups),
        "inner_splits": int(config["inner_splits"]),
        "base_models": base_models,
        "weight_step": float(config["weight_step"]),
        "selection_metric": config["selection_metric"],
        "random_state": int(config["random_state"]),
        "config_sha256": config_hash,
        "run_signature": run_signature,
        "elapsed_seconds": round(time.time() - started, 3),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "outputs": [
            str(archive_path),
            str(summary_path),
            str(by_target_path),
            str(parameter_path),
            str(checkpoint_path),
        ],
    }
    manifest_path = RESULT_DIR / (
        f"q1_run_manifest_{experiment_id}"
        + (f"_{suffix}" if suffix else "")
        + ".json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.read_csv(summary_path).to_string(index=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screening", "full"], required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.stage), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
