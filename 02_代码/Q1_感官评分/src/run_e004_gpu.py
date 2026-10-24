"""Run the independent CUDA cumulative XGBoost candidate for Q1."""

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
from ordered_models_gpu import fit_predict_cumulative_xgb
from q1_metrics import matrix_metrics
from second_round_runner import (
    load_run_checkpoint,
    load_config,
    make_inner_splits,
    outer_indices,
    repeats_for_stage,
    save_run_checkpoint,
    save_prediction_archive,
    write_metric_artifacts,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "E004_gpu_ordinal_xgb.yaml"


def _fit_profile(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    profile: dict,
    random_state: int,
) -> np.ndarray:
    return fit_predict_cumulative_xgb(
        x_train,
        y_train,
        x_validation,
        max_depth=int(profile["max_depth"]),
        min_child_weight=float(profile["min_child_weight"]),
        learning_rate=float(profile["learning_rate"]),
        n_estimators=int(profile["n_estimators"]),
        subsample=float(profile["subsample"]),
        colsample_bytree=float(profile["colsample_bytree"]),
        reg_lambda=float(profile["reg_lambda"]),
        random_state=random_state,
    ).continuous


def _selection_key(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        metrics["mae"],
        -metrics["spearman"],
        -metrics["qwk_proxy"],
        -metrics["acc_pm1_proxy"],
        -metrics["r2"],
    )


def run(stage: str) -> dict:
    started = time.time()
    config = load_config(CONFIG_PATH)
    experiment_id = str(config["experiment_id"])
    repeats = repeats_for_stage(config, stage)
    suffix = "screening" if stage == "screening" else ""
    x_frame, y_frame, style = load_q1()
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    splits = pd.read_csv(RESULT_DIR / "q1_cv_splits.csv")
    x = x_frame.to_numpy(float)
    y = y_frame.to_numpy(float)
    predictions = np.full((len(repeats), *y.shape), np.nan, dtype=float)
    fold_assignment = np.full((len(repeats), len(y)), -1, dtype=np.int8)
    parameter_rows: list[dict] = []
    config_hash = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    signature_payload = {
        "experiment_id": experiment_id,
        "stage": stage,
        "repeats": repeats,
        "config_sha256": config_hash,
        "cv_sha256": hashlib.sha256(
            (RESULT_DIR / "q1_cv_splits.csv").read_bytes()
        ).hexdigest(),
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
        profile_results = []
        for profile in config["profiles"]:
            inner_prediction = np.full_like(y_train, np.nan, dtype=float)
            for inner_fold, (inner_train, inner_validation) in enumerate(inner_splits):
                inner_prediction[inner_validation] = _fit_profile(
                    x_train[inner_train],
                    y_train[inner_train],
                    x_train[inner_validation],
                    profile,
                    random_state=split_seed + inner_fold * 1000,
                )
            if not np.isfinite(inner_prediction).all():
                raise RuntimeError("E004-GPU inner OOF predictions are incomplete")
            profile_results.append(
                {"profile": profile, "metrics": matrix_metrics(y_train, inner_prediction)}
            )
        selected = min(profile_results, key=lambda row: _selection_key(row["metrics"]))
        predictions[repeat_position, validation_index] = _fit_profile(
            x_train, y_train, x_validation, selected["profile"], random_state=split_seed
        )
        fold_assignment[repeat_position, validation_index] = fold
        parameter_rows.append(
            {
                "experiment_id": experiment_id,
                "stage": stage,
                "repeat": repeat,
                "fold": fold,
                "selected_profile": selected["profile"]["name"],
                "selected_profile_json": json.dumps(selected["profile"], sort_keys=True),
                "inner_metrics_json": json.dumps(
                    {row["profile"]["name"]: row["metrics"] for row in profile_results},
                    sort_keys=True,
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
            f"[{split_number:02d}/{len(selected_split_groups)}] repeat={repeat} "
            f"fold={fold} profile={selected['profile']['name']} "
            f"inner_mae={selected['metrics']['mae']:.4f}",
            flush=True,
        )

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
        predictions={"pred_ordinal_xgb": predictions},
    )
    summary_path, by_target_path = write_metric_artifacts(
        experiment_id, archive_path, ["pred_ordinal_xgb"], suffix
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
        "device": config["device"],
        "repeats": repeats,
        "outer_folds": len(selected_split_groups),
        "inner_splits": int(config["inner_splits"]),
        "profile_count": len(config["profiles"]),
        "selection_metric": config["selection_metric"],
        "preprocessing": "outer/inner-fold median + missing indicator",
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
