from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

try:
    style_adjustment = importlib.import_module("style_adjustment")
except ModuleNotFoundError:
    style_adjustment = None


class StyleAdjustmentTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(style_adjustment, "style_adjustment module is missing")
        return style_adjustment

    def test_training_style_means_are_removed_and_reused_for_validation(self):
        module = self.require_module()
        train = pd.DataFrame(
            {"x1": [1.0, 3.0, 10.0, 14.0], "x2": [2.0, 4.0, 20.0, 24.0]}
        )
        validation = pd.DataFrame({"x1": [5.0, 18.0], "x2": [8.0, 30.0]})
        train_style = pd.Series(["A", "A", "B", "B"])
        validation_style = pd.Series(["A", "B"])
        adjusted = module.fit_style_residualizer(
            train, validation, train_style, validation_style
        )
        np.testing.assert_allclose(
            adjusted.train.groupby(train_style.to_numpy()).mean().to_numpy(), 0.0
        )
        np.testing.assert_allclose(
            adjusted.validation.to_numpy(), [[3.0, 5.0], [6.0, 8.0]]
        )

    def test_unseen_validation_style_uses_training_global_mean_only(self):
        module = self.require_module()
        train = pd.DataFrame({"x": [0.0, 2.0, 10.0, 14.0]})
        validation = pd.DataFrame({"x": [100.0]})
        train_style = pd.Series(["A", "A", "B", "B"])
        validation_style = pd.Series(["unseen"])
        adjusted = module.fit_style_residualizer(
            train, validation, train_style, validation_style
        )
        np.testing.assert_allclose(adjusted.validation["x"], [93.5])
        np.testing.assert_allclose(adjusted.global_mean["x"], 6.5)

    def test_validation_changes_do_not_change_fitted_style_means(self):
        module = self.require_module()
        train = pd.DataFrame({"x": [1.0, 3.0, 10.0, 14.0]})
        style = pd.Series(["A", "A", "B", "B"])
        first = module.fit_style_residualizer(
            train,
            pd.DataFrame({"x": [5.0, 18.0]}),
            style,
            pd.Series(["A", "B"]),
        )
        second = module.fit_style_residualizer(
            train,
            pd.DataFrame({"x": [5000.0, -1800.0]}),
            style,
            pd.Series(["A", "B"]),
        )
        pd.testing.assert_frame_equal(first.style_means, second.style_means)
        pd.testing.assert_series_equal(first.global_mean, second.global_mean)

    def test_missing_values_remain_missing_until_fold_preprocessing(self):
        module = self.require_module()
        train = pd.DataFrame({"x": [1.0, np.nan, 10.0, 14.0]})
        validation = pd.DataFrame({"x": [np.nan, 18.0]})
        train_style = pd.Series(["A", "A", "B", "B"])
        validation_style = pd.Series(["A", "B"])
        adjusted = module.fit_style_residualizer(
            train, validation, train_style, validation_style
        )
        self.assertTrue(np.isnan(adjusted.train.iloc[1, 0]))
        self.assertTrue(np.isnan(adjusted.validation.iloc[0, 0]))


if __name__ == "__main__":
    unittest.main()
