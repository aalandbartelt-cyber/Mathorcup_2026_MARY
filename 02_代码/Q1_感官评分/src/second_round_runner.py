"""Shared frozen-split and artifact helpers for Q1 second-round experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from data_contract import RESULT_DIR
from q1_metrics import evaluate_prediction_matrix, summarise_target_metrics


def save_run_checkpoint(
    path: Path,
    *,
    run_signature: str,
    predictions: np.ndarray,
    fold_assignment: np.ndarray,
    parameter_rows: Sequence[Mapping],
) -> None:
    """Atomically persist a partial outer-OOF run for exact resumption."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            prefix=f".{target.stem}.",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(
                temporary,
                run_signature=np.asarray(str(run_signature)),
                predictions=np.asarray(predictions, dtype=float),
                fold_assignment=np.asarray(fold_assignment),
                parameter_rows_json=np.asarray(
                    json.dumps(list(parameter_rows), ensure_ascii=False, sort_keys=True)
                ),
            )
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_run_checkpoint(
    path: Path,
    *,
    run_signature: str,
    prediction_shape: Sequence[int],
    assignment_shape: Sequence[int],
) -> dict[str, object]:
    """Load and validate a partial run checkpoint against frozen inputs."""
    target = Path(path)
    with np.load(target, allow_pickle=False) as archive:
        saved_signature = str(archive["run_signature"].item())
        if saved_signature != str(run_signature):
            raise RuntimeError("checkpoint signature does not match frozen run")
        predictions = np.asarray(archive["predictions"], dtype=float).copy()
        fold_assignment = np.asarray(archive["fold_assignment"]).copy()
        parameter_rows = json.loads(str(archive["parameter_rows_json"].item()))
    if predictions.shape != tuple(prediction_shape):
        raise RuntimeError("checkpoint prediction shape does not match frozen run")
    if fold_assignment.shape != tuple(assignment_shape):
        raise RuntimeError("checkpoint assignment shape does not match frozen run")
    if not isinstance(parameter_rows, list):
        raise RuntimeError("checkpoint parameter rows are invalid")
    return {
        "predictions": predictions,
        "fold_assignment": fold_assignment,
        "parameter_rows": parameter_rows,
    }


def load_config(path: Path) -> dict:
    """Load one frozen YAML experiment configuration."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "experiment_id" not in config:
        raise ValueError(f"invalid experiment config: {path}")
    return config


def parse_repeats(text: str, available: Sequence[int]) -> list[int]:
    """Parse a comma-separated repeat list and reject unavailable values."""
    try:
        selected = sorted(
            {int(value.strip()) for value in text.split(",") if value.strip()}
        )
    except ValueError as error:
        raise ValueError("repeat list must contain integers") from error
    if not selected or not set(selected).issubset(set(available)):
        raise ValueError("requested repeats are not available")
    return selected


def repeats_for_stage(config: Mapping, stage: str) -> list[int]:
    """Return the pre-registered repeat IDs for screening or full execution."""
    key = f"{stage}_repeats"
    if stage not in {"screening", "full"} or key not in config:
        raise ValueError("stage must be screening or full")
    return parse_repeats(",".join(map(str, config[key])), config["full_repeats"])


def outer_indices(
    splits: pd.DataFrame,
    repeat: int,
    fold: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint train/validation indices for one frozen outer fold."""
    validation = splits.loc[
        splits["repeat"].eq(repeat) & splits["fold"].eq(fold),
        "row_position",
    ].to_numpy(int)
    if len(validation) == 0 or len(np.unique(validation)) != len(validation):
        raise ValueError("outer validation indices are empty or duplicated")
    if validation.min() < 0 or validation.max() >= n_samples:
        raise ValueError("outer validation index outside sample range")
    mask = np.zeros(n_samples, dtype=bool)
    mask[validation] = True
    train = np.flatnonzero(~mask)
    if len(train) + len(validation) != n_samples:
        raise RuntimeError("outer split does not cover every sample exactly once")
    return train, validation


def make_inner_splits(
    n_samples: int,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic inner folds on outer-training row positions."""
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.arange(n_samples)))


def assert_complete_oof(prediction: np.ndarray) -> None:
    """Reject incomplete, non-finite, or incorrectly shaped OOF tensors."""
    array = np.asarray(prediction, dtype=float)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise RuntimeError("incomplete or non-finite OOF predictions")


def save_prediction_archive(
    path: Path,
    *,
    y_true: np.ndarray,
    beer: np.ndarray,
    style: np.ndarray,
    target_code: np.ndarray,
    repeats: np.ndarray,
    fold_assignment: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> None:
    """Write one self-describing compressed OOF archive."""
    y_array = np.asarray(y_true, dtype=float)
    repeat_array = np.asarray(repeats, dtype=int)
    assignment = np.asarray(fold_assignment)
    if assignment.shape != (len(repeat_array), len(y_array)):
        raise ValueError("fold assignment shape does not match repeats and samples")
    for name, value in predictions.items():
        assert_complete_oof(value)
        if np.asarray(value).shape != (
            len(repeat_array),
            *y_array.shape,
        ):
            raise ValueError(f"{name} prediction shape does not match archive metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y_true=y_array,
        beer=np.asarray(beer),
        style=np.asarray(style),
        target_code=np.asarray(target_code),
        repeat=repeat_array,
        fold_assignment=assignment,
        **predictions,
    )


def write_metric_artifacts(
    experiment_id: str,
    archive_path: Path,
    prediction_keys: Sequence[str],
    suffix: str,
) -> tuple[Path, Path]:
    """Evaluate every prediction tensor and write target/summary CSV files."""
    archive = np.load(archive_path, allow_pickle=True)
    y_true = np.asarray(archive["y_true"], dtype=float)
    targets = np.asarray(archive["target_code"])
    repeats = np.asarray(archive["repeat"], dtype=int)
    frames = []
    for key in prediction_keys:
        if key not in archive.files:
            raise KeyError(f"prediction key missing from archive: {key}")
        prediction = np.asarray(archive[key], dtype=float)
        assert_complete_oof(prediction)
        for position, repeat in enumerate(repeats):
            frame = evaluate_prediction_matrix(
                y_true,
                prediction[position],
                targets,
                experiment_id=experiment_id,
                evaluation=f"{key}:repeat_{repeat}",
            )
            frame.insert(2, "model", key.removeprefix("pred_"))
            frames.append(frame)
        mean_frame = evaluate_prediction_matrix(
            y_true,
            prediction.mean(axis=0),
            targets,
            experiment_id=experiment_id,
            evaluation=f"{key}:mean_oof",
        )
        mean_frame.insert(2, "model", key.removeprefix("pred_"))
        frames.append(mean_frame)
    by_target = pd.concat(frames, ignore_index=True)
    stem = f"_{suffix}" if suffix else ""
    by_target_path = (
        RESULT_DIR / f"q1_extended_metrics_by_target_{experiment_id}{stem}.csv"
    )
    summary_path = RESULT_DIR / f"q1_extended_metrics_{experiment_id}{stem}.csv"
    by_target.to_csv(by_target_path, index=False)
    summarise_target_metrics(by_target).to_csv(summary_path, index=False)
    return summary_path, by_target_path
