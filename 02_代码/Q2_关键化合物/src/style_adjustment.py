"""Leakage-safe removal of tasting-style mean effects for Q2 sensitivity tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StyleResidualized:
    train: pd.DataFrame
    validation: pd.DataFrame
    style_means: pd.DataFrame
    global_mean: pd.Series


def _as_style(style: pd.Series | np.ndarray, n_rows: int) -> pd.Series:
    values = pd.Series(np.asarray(style, dtype=object), dtype="object")
    if len(values) != int(n_rows):
        raise ValueError("style labels and data rows disagree")
    if values.isna().any():
        raise ValueError("style labels must be complete")
    return values.astype(str)


def fit_style_residualizer(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_style: pd.Series | np.ndarray,
    validation_style: pd.Series | np.ndarray,
) -> StyleResidualized:
    """Fit style means on training rows and subtract them from both partitions.

    A validation style unseen during fitting uses the training global mean.  No
    missing value is imputed here; the downstream fold-local preprocessor owns
    that operation and its missingness indicators.
    """
    train_frame = pd.DataFrame(train).reset_index(drop=True).astype(float)
    validation_frame = pd.DataFrame(validation).reset_index(drop=True).astype(float)
    if train_frame.columns.tolist() != validation_frame.columns.tolist():
        raise ValueError("training and validation columns disagree")
    training_style = _as_style(train_style, len(train_frame))
    validation_labels = _as_style(validation_style, len(validation_frame))

    global_mean = train_frame.mean(axis=0, skipna=True).fillna(0.0)
    grouped = train_frame.groupby(training_style.to_numpy(), sort=True).mean()
    style_means = grouped.fillna(global_mean)

    def offsets(labels: pd.Series) -> pd.DataFrame:
        rows = []
        for label in labels:
            rows.append(
                style_means.loc[label].to_numpy(float)
                if label in style_means.index
                else global_mean.to_numpy(float)
            )
        return pd.DataFrame(rows, columns=train_frame.columns)

    train_adjusted = train_frame - offsets(training_style)
    validation_adjusted = validation_frame - offsets(validation_labels)
    return StyleResidualized(
        train=train_adjusted,
        validation=validation_adjusted,
        style_means=style_means,
        global_mean=global_mean,
    )
