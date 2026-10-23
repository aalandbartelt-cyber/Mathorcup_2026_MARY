from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from baseline_models import (  # noqa: E402
    fit_predict_global_mean,
    fit_predict_pls,
    fit_predict_ridge,
    fit_predict_shrunk_style,
    regression_metrics,
)


class BaselineModelTests(unittest.TestCase):
    def test_global_mean_uses_training_targets_only(self) -> None:
        y_train = np.array([[0.0, 2.0], [2.0, 4.0]])
        prediction = fit_predict_global_mean(y_train, n_validation=2)
        np.testing.assert_allclose(prediction, [[1.0, 3.0], [1.0, 3.0]])

    def test_shrunk_style_falls_back_to_global_mean_for_unseen_style(self) -> None:
        y_train = np.array([[0.0], [2.0], [8.0]])
        train_style = pd.Series(["A", "A", "B"])
        validation_style = pd.Series(["A", "C"])
        prediction = fit_predict_shrunk_style(
            y_train,
            train_style,
            validation_style,
            shrinkage=1.0,
        )
        global_mean = 10.0 / 3.0
        expected_a = (2 * 1.0 + global_mean) / 3.0
        np.testing.assert_allclose(prediction[:, 0], [expected_a, global_mean])

    def test_regression_metrics_reports_negative_r2_without_clipping(self) -> None:
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        metrics = regression_metrics(y_true, y_pred)
        self.assertLess(metrics["r2"], 0)
        self.assertEqual(metrics["mae"], 1.0)
        self.assertGreater(metrics["rmse"], 1.0)

    def test_ridge_pipeline_handles_missing_values_and_multiple_targets(self) -> None:
        x_train = np.array([[0.0, 1.0], [1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
        y_train = np.column_stack([x_train[:, 0], 2 * x_train[:, 0]])
        x_validation = np.array([[1.5, np.nan], [4.0, 5.0]])
        prediction = fit_predict_ridge(x_train, y_train, x_validation, alpha=1.0)
        self.assertEqual(prediction.shape, (2, 2))
        self.assertTrue(np.isfinite(prediction).all())

    def test_pls_pipeline_handles_missing_values_and_multiple_targets(self) -> None:
        x_train = np.array(
            [[0.0, 1.0], [1.0, np.nan], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        )
        y_train = np.column_stack([x_train[:, 0], 2 * x_train[:, 0]])
        x_validation = np.array([[1.5, np.nan], [5.0, 6.0]])
        prediction = fit_predict_pls(
            x_train,
            y_train,
            x_validation,
            n_components=1,
        )
        self.assertEqual(prediction.shape, (2, 2))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()

