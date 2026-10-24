"""Compute continuous and ordered-proxy metrics for frozen E003 OOF arrays."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data_contract import RESULT_DIR
from q1_metrics import (
    evaluate_prediction_matrix,
    ordered_proxy_metrics,
    quantile_levels,
    summarise_target_metrics,
    target_metrics,
)


EXPERIMENT_ID = "E003"
SOURCE = RESULT_DIR / "q1_oof_predictions_E003.npz"


def main() -> None:
    archive = np.load(SOURCE, allow_pickle=True)
    y_true = np.asarray(archive["y_true"], dtype=float)
    predictions = np.asarray(archive["pred_extra_trees"], dtype=float)
    target_codes = np.asarray(archive["target_code"])
    if predictions.ndim != 3 or predictions.shape[1:] != y_true.shape:
        raise ValueError("unexpected E003 prediction tensor shape")
    if not np.isfinite(predictions).all():
        raise ValueError("E003 OOF predictions contain non-finite values")

    evaluations = [
        evaluate_prediction_matrix(
            y_true,
            predictions[repeat],
            target_codes,
            experiment_id=EXPERIMENT_ID,
            evaluation=f"repeat_{repeat}",
        )
        for repeat in range(predictions.shape[0])
    ]
    evaluations.append(
        evaluate_prediction_matrix(
            y_true,
            predictions.mean(axis=0),
            target_codes,
            experiment_id=EXPERIMENT_ID,
            evaluation="mean_oof",
        )
    )
    by_target = pd.concat(evaluations, ignore_index=True)
    summary = summarise_target_metrics(by_target)

    by_target_path = RESULT_DIR / "q1_extended_metrics_by_target_E003.csv"
    summary_path = RESULT_DIR / "q1_extended_metrics_E003.csv"
    manifest_path = RESULT_DIR / "q1_extended_metrics_manifest_E003.json"
    by_target.to_csv(by_target_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "source": str(SOURCE.relative_to(RESULT_DIR.parents[1])),
        "prediction_shape": list(predictions.shape),
        "continuous_metrics": ["R2", "MAE", "RMSE", "Spearman"],
        "ordered_proxy_metrics": [
            "QWK", "ACC", "ACC_pm1", "LevelMAE", "level_Spearman"
        ],
        "ordered_proxy_definition": (
            "For each target, observed values and predictions are independently "
            "mapped to integer levels 0..6 using their own full-sample empirical "
            "quantiles; target scores are then macro-averaged."
        ),
        "usage_limit": (
            "Post-hoc BZD-compatible comparison only. This mapping is not used "
            "for final 0--6 inference because it uses the full OOF distribution."
        ),
        "outputs": [str(summary_path), str(by_target_path)],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
