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
    fit_predict_extra_trees,
    fit_predict_pls,
    fit_predict_pls_quantile,
    fit_predict_ridge,
    fit_predict_ridge_quantile,
    fit_predict_shrunk_style,
    empirical_cdf_grade,
    regression_metrics,
    select_parameter_by_mae,
)


class BaselineModelTests(unittest.TestCase):
    def test_empirical_cdf_grade_uses_only_reference_distribution(self) -> None:
        reference = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        prediction = np.array([[-1.0, 20.0], [1.0, 40.0]])
        grade = empirical_cdf_grade(reference, prediction)
        np.testing.assert_allclose(grade, [[0.0, 4.0], [4.0, 6.0]])
        self.assertTrue(((grade >= 0.0) & (grade <= 6.0)).all())

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

    def test_inner_selection_minimizes_validation_mae(self) -> None:
        y = np.arange(9, dtype=float).reshape(-1, 1)

        def predict_fold(train_index, validation_index, candidate):
            del train_index
            if candidate == "oracle":
                return y[validation_index]
            return np.zeros((len(validation_index), 1))

        selected, scores = select_parameter_by_mae(
            targets=y,
            candidates=["zero", "oracle"],
            predict_fold=predict_fold,
            n_splits=3,
            random_state=7,
        )
        self.assertEqual(selected, "oracle")
        self.assertEqual(scores["oracle"], 0.0)
        self.assertGreater(scores["zero"], 0.0)

    def test_quantile_ridge_remains_finite_for_extreme_validation_value(self) -> None:
        x_train = np.column_stack([np.arange(12, dtype=float), np.arange(12, dtype=float)])
        y_train = np.column_stack([x_train[:, 0], 2 * x_train[:, 0]])
        x_validation = np.array([[100000.0, np.nan]])
        prediction = fit_predict_ridge_quantile(
            x_train,
            y_train,
            x_validation,
            alpha=10.0,
            random_state=7,
        )
        self.assertEqual(prediction.shape, (1, 2))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(np.abs(prediction).max(), 100.0)

    def test_quantile_pls_remains_finite_for_extreme_validation_value(self) -> None:
        x_train = np.column_stack([np.arange(12, dtype=float), np.arange(12, dtype=float)])
        y_train = np.column_stack([x_train[:, 0], 2 * x_train[:, 0]])
        x_validation = np.array([[100000.0, np.nan]])
        prediction = fit_predict_pls_quantile(
            x_train,
            y_train,
            x_validation,
            n_components=1,
            random_state=7,
        )
        self.assertEqual(prediction.shape, (1, 2))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(np.abs(prediction).max(), 100.0)

    def test_extra_trees_handles_missing_values_and_is_deterministic(self) -> None:
        x_train = np.array(
            [[0.0, 1.0], [1.0, np.nan], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        )
        y_train = np.column_stack([x_train[:, 0], 2 * x_train[:, 0]])
        x_validation = np.array([[1.5, np.nan], [5.0, 6.0]])
        first = fit_predict_extra_trees(
            x_train,
            y_train,
            x_validation,
            n_estimators=20,
            max_features=1.0,
            min_samples_leaf=1,
            max_depth=None,
            random_state=7,
        )
        second = fit_predict_extra_trees(
            x_train,
            y_train,
            x_validation,
            n_estimators=20,
            max_features=1.0,
            min_samples_leaf=1,
            max_depth=None,
            random_state=7,
        )
        self.assertEqual(first.shape, (2, 2))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()

