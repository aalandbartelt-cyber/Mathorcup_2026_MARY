"""Run conditional E007 targetwise route weights with shared shrinkage."""

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
    apply_targetwise_blend,
    select_feasible_convex_blend,
    shrink_target_weights,
)
from q1_metrics import matrix_metrics
from run_e004_ordinal_hgb import _fit_profile as fit_ordinal_profile
from run_e006_dual_route import (
    _fit_selected_continuous_route,
    _screening_decision,
    _select_continuous_route,
    _select_ordinal_route,
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


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "E007_shrunk_gating.yaml"
CONFIG_DIR = CONFIG_PATH.parent


def _e006_passes_hard_thresholds(stage: str, thresholds: dict) -> bool:
    suffix = "_screening" if stage == "screening" else ""
    path = RESULT_DIR / f"q1_extended_metrics_E006{suffix}.csv"
    if not path.exists():
        raise RuntimeError(f"E007 requires completed E006 metrics: {path}")
    frame = pd.read_csv(path)
    rows = frame.loc[
        frame["evaluation"].str.startswith("pred_dual_route:repeat_")
    ]
    if rows.empty:
        raise RuntimeError("E006 has no repeat-level dual-route metrics")
    mean = rows[
        ["r2", "mae", "spearman", "qwk_proxy", "acc_pm1_proxy"]
    ].mean()
    return bool(
        mean["r2"] > float(thresholds["r2_min"])
        and mean["mae"] < float(thresholds["mae_max"])
        and mean["spearman"] > float(thresholds["spearman_min"])
        and mean["qwk_proxy"] > float(thresholds["qwk_proxy_min"])
        and mean["acc_pm1_proxy"] > float(thresholds["acc_pm1_proxy_min"])
    )


def _select_shrunk_weights(
    y_true: np.ndarray,
    continuous_prediction: np.ndarray,
    ordinal_prediction: np.ndarray,
    route_candidates: np.ndarray,
    shrinkage_strengths: list[float],
    r2_drop: float,
    mae_increase: float,
) -> tuple[np.ndarray, float, dict]:
    route_stack = np.stack([continuous_prediction, ordinal_prediction], axis=-1)
    continuous_metrics = matrix_metrics(y_true, continuous_prediction)
    global_selection = select_feasible_convex_blend(
        y_true,
        route_stack,
        route_candidates,
        r2_floor=continuous_metrics["r2"] - r2_drop,
        mae_ceiling=continuous_metrics["mae"] + mae_increase,
    )
    target_weights = np.empty((y_true.shape[1], route_stack.shape[-1]), dtype=float)
    target_metrics = []
    for target_position in range(y_true.shape[1]):
        target_continuous_metrics = matrix_metrics(
            y_true[:, [target_position]], continuous_prediction[:, [target_position]]
        )
        selection = select_feasible_convex_blend(
            y_true[:, [target_position]],
            route_stack[:, [target_position], :],
            route_candidates,
            r2_floor=target_continuous_metrics["r2"] - r2_drop,
            mae_ceiling=target_continuous_metrics["mae"] + mae_increase,
        )
        target_weights[target_position] = selection.weights
        target_metrics.append(selection.metrics)

    scored = []
    for order, shrinkage in enumerate(shrinkage_strengths):
        weights = shrink_target_weights(
            target_weights, global_selection.weights, float(shrinkage)
        )
        prediction = apply_targetwise_blend(route_stack, weights)
        metrics = matrix_metrics(y_true, prediction)
        feasible = (
            metrics["r2"] >= continuous_metrics["r2"] - r2_drop
            and metrics["mae"] <= continuous_metrics["mae"] + mae_increase
        )
        violation = max(
            0.0, continuous_metrics["r2"] - r2_drop - metrics["r2"]
        ) + max(0.0, metrics["mae"] - continuous_metrics["mae"] - mae_increase)
        rank = (
            metrics["spearman"],
            metrics["qwk_proxy"],
            metrics["acc_pm1_proxy"],
            metrics["r2"],
            -metrics["mae"],
        )
        scored.append(
            {
                "order": order,
                "shrinkage": float(shrinkage),
                "weights": weights,
                "metrics": metrics,
                "feasible": feasible,
                "violation": violation,
                "rank": rank,
            }
        )
    feasible_rows = [row for row in scored if row["feasible"]]
    if feasible_rows:
        selected = max(
            feasible_rows, key=lambda row: (*row["rank"], -row["order"])
        )
    else:
        selected = min(
            scored,
            key=lambda row: (
                row["violation"],
                *(-value for value in row["rank"]),
                row["order"],
            ),
        )
    detail = {
        "global_weights": global_selection.weights.tolist(),
        "global_metrics": global_selection.metrics,
        "target_weights": target_weights.tolist(),
        "target_metrics": target_metrics,
        "shrinkage_candidates": [
            {
                "shrinkage": row["shrinkage"],
                "metrics": row["metrics"],
                "feasible": row["feasible"],
            }
            for row in scored
        ],
        "selected_metrics": selected["metrics"],
        "selected_feasible": selected["feasible"],
    }
    return selected["weights"], selected["shrinkage"], detail


def run(stage: str) -> dict:
    started = time.time()
    config = load_config(CONFIG_PATH)
    e006_config_path = CONFIG_DIR / config["source_config"]
    e006_config = load_config(e006_config_path)
    ordinal_config_path = CONFIG_DIR / e006_config["source_configs"]["ordinal"]
    continuous_config_path = CONFIG_DIR / e006_config["source_configs"]["continuous"]
    ordinal_config = load_config(ordinal_config_path)
    continuous_config = load_config(continuous_config_path)
    experiment_id = str(config["experiment_id"])
    repeats = repeats_for_stage(config, stage)
    suffix = "screening" if stage == "screening" else ""

    if _e006_passes_hard_thresholds(stage, e006_config["hard_thresholds"]):
        manifest = {
            "experiment_id": experiment_id,
            "stage": stage,
            "status": "skipped",
            "reason": "E006 passes all five hard thresholds",
            "entry_condition": config["entry_condition"],
        }
        manifest_path = RESULT_DIR / (
            f"q1_run_manifest_{experiment_id}"
            + (f"_{suffix}" if suffix else "")
            + ".json"
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest

    x_frame, y_frame, style = load_q1()
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    splits_path = RESULT_DIR / "q1_cv_splits.csv"
    splits = pd.read_csv(splits_path)
    x = x_frame.to_numpy(float)
    y = y_frame.to_numpy(float)
    output_models = ["continuous", "ordinal", "shrunk_gate"]
    predictions = np.full(
        (len(repeats), len(y), y.shape[1], len(output_models)), np.nan, dtype=float
    )
    fold_assignment = np.full((len(repeats), len(y)), -1, dtype=np.int8)
    parameter_rows: list[dict] = []
    config_paths = [CONFIG_PATH, e006_config_path, ordinal_config_path, continuous_config_path]
    config_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in config_paths
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
    route_candidates = np.asarray(
        [
            [1.0 - float(share), float(share)]
            for share in e006_config["ordinal_shares"]
        ]
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
        selected_target_weights, shrinkage, shrinkage_detail = _select_shrunk_weights(
            y_train,
            continuous_inner,
            ordinal_inner,
            route_candidates,
            [float(value) for value in config["shrinkage_strengths"]],
            r2_drop=float(tolerance["r2_drop"]),
            mae_increase=float(tolerance["mae_increase"]),
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
        shrunk_outer = apply_targetwise_blend(
            np.stack([continuous_outer, ordinal_outer], axis=-1),
            selected_target_weights,
        )
        predictions[repeat_position, validation_index, :, 0] = continuous_outer
        predictions[repeat_position, validation_index, :, 1] = ordinal_outer
        predictions[repeat_position, validation_index, :, 2] = shrunk_outer
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
                "selected_shrinkage": shrinkage,
                "target_weights_json": json.dumps(selected_target_weights.tolist()),
                "shrinkage_detail_json": json.dumps(
                    shrinkage_detail, sort_keys=True
                ),
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
            f"repeat={repeat} fold={fold} shrinkage={shrinkage:.2f}",
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
        _screening_decision(summary_path, "pred_shrunk_gate")
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
        "shrinkage_strengths": config["shrinkage_strengths"],
        "entry_condition": config["entry_condition"],
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
