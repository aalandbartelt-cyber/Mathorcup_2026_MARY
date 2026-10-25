"""Run E006 nested continuous--ordinal blending on frozen Q1 folds."""

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

from data_contract import ATTACHMENTS, RESULT_DIR, load_q1
from ensemble_models import (
    apply_convex_blend,
    select_feasible_convex_blend,
    simplex_weight_grid,
)
from q1_metrics import matrix_metrics
from run_e004_ordinal_hgb import _fit_profile as fit_ordinal_profile
from run_e005_continuous_ensemble import (
    _candidates as continuous_candidates,
    _fit_base as fit_continuous_base,
    _parameter_key,
)
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


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "E006_dual_route.yaml"
CONFIG_DIR = CONFIG_PATH.parent


def _select_ordinal_route(
    x_train: np.ndarray,
    y_train: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    config: dict,
    split_seed: int,
) -> tuple[dict, np.ndarray, dict[str, dict[str, float]]]:
    profile_rows = []
    for profile_position, profile in enumerate(config["profiles"]):
        prediction = np.full_like(y_train, np.nan, dtype=float)
        for inner_fold, (inner_train, inner_validation) in enumerate(inner_splits):
            prediction[inner_validation] = fit_ordinal_profile(
                x_train[inner_train],
                y_train[inner_train],
                x_train[inner_validation],
                profile,
                random_state=split_seed + profile_position * 1_000 + inner_fold,
            )
        if not np.isfinite(prediction).all():
            raise RuntimeError("E006 ordinal inner OOF is incomplete")
        profile_rows.append(
            {
                "profile": profile,
                "prediction": prediction,
                "metrics": matrix_metrics(y_train, prediction),
            }
        )
    selected = min(profile_rows, key=lambda row: _parameter_key(row["metrics"]))
    metric_log = {
        str(row["profile"]["name"]): row["metrics"] for row in profile_rows
    }
    return selected["profile"], selected["prediction"], metric_log


def _select_continuous_route(
    x_train: np.ndarray,
    y_train: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    config: dict,
    split_seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, dict]:
    base_models = list(config["base_models"])
    selected_parameters: dict[str, object] = {}
    selected_predictions: dict[str, np.ndarray] = {}
    candidate_metric_log: dict[str, list[dict]] = {}
    for model_position, model in enumerate(base_models):
        candidate_rows = []
        for candidate_position, candidate in enumerate(
            continuous_candidates(config, model)
        ):
            prediction = np.full_like(y_train, np.nan, dtype=float)
            for inner_fold, (inner_train, inner_validation) in enumerate(inner_splits):
                prediction[inner_validation] = fit_continuous_base(
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
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"E006 {model} inner OOF is incomplete")
            candidate_rows.append(
                {
                    "parameter": candidate,
                    "prediction": prediction,
                    "metrics": matrix_metrics(y_train, prediction),
                }
            )
        selected = min(candidate_rows, key=lambda row: _parameter_key(row["metrics"]))
        selected_parameters[model] = selected["parameter"]
        selected_predictions[model] = selected["prediction"]
        candidate_metric_log[model] = [
            {"parameter": row["parameter"], "metrics": row["metrics"]}
            for row in candidate_rows
        ]

    base_stack = np.stack(
        [selected_predictions[model] for model in base_models], axis=-1
    )
    base_metrics = {
        model: matrix_metrics(y_train, selected_predictions[model])
        for model in base_models
    }
    tolerance = config["continuous_tolerance"]
    continuous_blend = select_feasible_convex_blend(
        y_train,
        base_stack,
        simplex_weight_grid(len(base_models), float(config["weight_step"])),
        r2_floor=max(row["r2"] for row in base_metrics.values())
        - float(tolerance["r2_drop"]),
        mae_ceiling=min(row["mae"] for row in base_metrics.values())
        + float(tolerance["mae_increase"]),
    )
    prediction = apply_convex_blend(base_stack, continuous_blend.weights)
    detail = {
        "candidate_metrics": candidate_metric_log,
        "base_metrics": base_metrics,
        "blend_metrics": continuous_blend.metrics,
        "blend_feasible": continuous_blend.feasible,
    }
    return selected_parameters, continuous_blend.weights, prediction, detail


def _fit_selected_continuous_route(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    config: dict,
    selected_parameters: dict[str, object],
    weights: np.ndarray,
    split_seed: int,
) -> np.ndarray:
    predictions = []
    for model_position, model in enumerate(config["base_models"]):
        predictions.append(
            fit_continuous_base(
                model,
                selected_parameters[model],
                x_train,
                y_train,
                x_validation,
                config,
                random_state=split_seed + model_position * 100_000,
            )
        )
    return apply_convex_blend(np.stack(predictions, axis=-1), weights)


def _screening_decision(summary_path: Path, prediction_key: str) -> dict:
    candidate = pd.read_csv(summary_path)
    baseline = pd.read_csv(RESULT_DIR / "q1_extended_metrics_E003.csv")
    candidate = candidate.loc[
        candidate["evaluation"].str.startswith(f"{prediction_key}:repeat_")
    ].copy()
    repeat_ids = sorted(
        int(value.rsplit("_", 1)[1]) for value in candidate["evaluation"]
    )
    baseline = baseline.loc[
        baseline["evaluation"].isin([f"repeat_{value}" for value in repeat_ids])
    ]
    candidate_mean = candidate[
        ["r2", "mae", "spearman", "qwk_proxy", "acc_pm1_proxy"]
    ].mean()
    baseline_mean = baseline[
        ["r2", "mae", "spearman", "qwk_proxy", "acc_pm1_proxy"]
    ].mean()
    improved = {
        "r2": bool(candidate_mean["r2"] > baseline_mean["r2"]),
        "mae": bool(candidate_mean["mae"] < baseline_mean["mae"]),
        "spearman": bool(candidate_mean["spearman"] > baseline_mean["spearman"]),
        "qwk_proxy": bool(candidate_mean["qwk_proxy"] > baseline_mean["qwk_proxy"]),
        "acc_pm1_proxy": bool(
            candidate_mean["acc_pm1_proxy"] > baseline_mean["acc_pm1_proxy"]
        ),
    }
    promote = (
        sum(improved.values()) >= 4
        and (improved["spearman"] or improved["acc_pm1_proxy"])
        and candidate_mean["r2"] >= baseline_mean["r2"] - 0.002
        and candidate_mean["mae"] <= baseline_mean["mae"] + 0.001
    )
    return {
        "repeat_ids": repeat_ids,
        "candidate_mean": {key: float(value) for key, value in candidate_mean.items()},
        "baseline_mean": {key: float(value) for key, value in baseline_mean.items()},
        "improved": improved,
        "improved_count": int(sum(improved.values())),
        "promote_to_full": bool(promote),
    }


def run(stage: str) -> dict:
    started = time.time()
    config = load_config(CONFIG_PATH)
    ordinal_config_path = CONFIG_DIR / config["source_configs"]["ordinal"]
    continuous_config_path = CONFIG_DIR / config["source_configs"]["continuous"]
    ordinal_config = load_config(ordinal_config_path)
    continuous_config = load_config(continuous_config_path)
    experiment_id = str(config["experiment_id"])
    repeats = repeats_for_stage(config, stage)
    suffix = "screening" if stage == "screening" else ""

    x_frame, y_frame, style = load_q1()
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    splits_path = RESULT_DIR / "q1_cv_splits.csv"
    splits = pd.read_csv(splits_path)
    x = x_frame.to_numpy(float)
    y = y_frame.to_numpy(float)
    output_models = ["continuous", "ordinal", "dual_route"]
    predictions = np.full(
        (len(repeats), len(y), y.shape[1], len(output_models)), np.nan, dtype=float
    )
    fold_assignment = np.full((len(repeats), len(y)), -1, dtype=np.int8)
    parameter_rows: list[dict] = []

    config_hashes = {
        "e006": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "ordinal": hashlib.sha256(ordinal_config_path.read_bytes()).hexdigest(),
        "continuous": hashlib.sha256(continuous_config_path.read_bytes()).hexdigest(),
    }
    signature_payload = {
        "experiment_id": experiment_id,
        "stage": stage,
        "repeats": repeats,
        "config_sha256": config_hashes,
        "cv_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "chemistry_sha256": hashlib.sha256(
            (ATTACHMENTS / "1.csv").read_bytes()
        ).hexdigest(),
        "target_sha256": hashlib.sha256(
            (ATTACHMENTS / "4.csv").read_bytes()
        ).hexdigest(),
        "shape": list(y.shape),
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
    route_weights = np.asarray(
        [[1.0 - float(share), float(share)] for share in config["ordinal_shares"]]
    )
    tolerance = config["continuous_tolerance"]

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
        ordinal_profile, ordinal_inner, ordinal_metric_log = _select_ordinal_route(
            x_train, y_train, inner_splits, ordinal_config, split_seed
        )
        (
            continuous_parameters,
            continuous_weights,
            continuous_inner,
            continuous_detail,
        ) = _select_continuous_route(
            x_train, y_train, inner_splits, continuous_config, split_seed
        )
        continuous_inner_metrics = matrix_metrics(y_train, continuous_inner)
        route_stack = np.stack([continuous_inner, ordinal_inner], axis=-1)
        route_blend = select_feasible_convex_blend(
            y_train,
            route_stack,
            route_weights,
            r2_floor=continuous_inner_metrics["r2"]
            - float(tolerance["r2_drop"]),
            mae_ceiling=continuous_inner_metrics["mae"]
            + float(tolerance["mae_increase"]),
        )

        continuous_outer = _fit_selected_continuous_route(
            x_train,
            y_train,
            x_validation,
            continuous_config,
            continuous_parameters,
            continuous_weights,
            split_seed,
        )
        ordinal_outer = fit_ordinal_profile(
            x_train,
            y_train,
            x_validation,
            ordinal_profile,
            random_state=split_seed,
        )
        dual_outer = apply_convex_blend(
            np.stack([continuous_outer, ordinal_outer], axis=-1), route_blend.weights
        )
        predictions[repeat_position, validation_index, :, 0] = continuous_outer
        predictions[repeat_position, validation_index, :, 1] = ordinal_outer
        predictions[repeat_position, validation_index, :, 2] = dual_outer
        fold_assignment[repeat_position, validation_index] = fold
        parameter_rows.append(
            {
                "experiment_id": experiment_id,
                "stage": stage,
                "repeat": repeat,
                "fold": fold,
                "ordinal_profile_json": json.dumps(ordinal_profile, sort_keys=True),
                "ordinal_inner_metrics_json": json.dumps(
                    ordinal_metric_log, sort_keys=True
                ),
                "continuous_parameters_json": json.dumps(
                    continuous_parameters, sort_keys=True
                ),
                "continuous_weights_json": json.dumps(continuous_weights.tolist()),
                "continuous_detail_json": json.dumps(
                    continuous_detail, sort_keys=True
                ),
                "route_weights_json": json.dumps(route_blend.weights.tolist()),
                "route_metrics_json": json.dumps(route_blend.metrics, sort_keys=True),
                "route_feasible": route_blend.feasible,
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
            f"repeat={repeat} fold={fold} ordinal_share="
            f"{route_blend.weights[1]:.2f} feasible={route_blend.feasible}",
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
    screening_decision = (
        _screening_decision(summary_path, "pred_dual_route")
        if stage == "screening"
        else None
    )
    manifest = {
        "experiment_id": experiment_id,
        "stage": stage,
        "status": "complete",
        "repeats": repeats,
        "outer_folds": len(selected_split_groups),
        "inner_splits": int(config["inner_splits"]),
        "ordinal_shares": config["ordinal_shares"],
        "hard_thresholds": config["hard_thresholds"],
        "screening_decision": screening_decision,
        "random_state": int(config["random_state"]),
        "config_sha256": config_hashes,
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
