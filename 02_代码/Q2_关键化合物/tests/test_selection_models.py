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
    models = importlib.import_module("selection_models")
except ModuleNotFoundError:
    models = None


class SelectionModelTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(models, "selection_models module has not been implemented")
        return models

    def test_fold_preprocessor_fits_training_rows_only(self) -> None:
        module = self.require_module()
        train = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, np.nan, 3.0]})
        valid = pd.DataFrame({"a": [100.0], "b": [np.nan]})
        first = module.fit_fold_preprocessor(train, valid)
        changed = valid.copy()
        changed.loc[0] = [-9999.0, 8888.0]
        second = module.fit_fold_preprocessor(train, changed)
        np.testing.assert_allclose(first.x_train, second.x_train)
        np.testing.assert_allclose(first.medians, [1.0, 2.0])
        self.assertEqual(first.n_original_features, 2)
        self.assertEqual(first.x_train.shape[1], 3)

    def test_fold_preprocessor_winsorizes_from_training_bounds(self) -> None:
        module = self.require_module()
        train = pd.DataFrame(
            {"a": np.arange(10, dtype=float), "b": np.linspace(-2.0, 2.0, 10)}
        )
        valid = pd.DataFrame({"a": [1_000_000.0], "b": [-1_000_000.0]})
        fitted = module.fit_fold_preprocessor(
            train, valid, winsor_quantiles=(0.1, 0.9)
        )
        np.testing.assert_allclose(fitted.lower_bounds, [0.9, -1.6])
        np.testing.assert_allclose(fitted.upper_bounds, [8.1, 1.6])
        self.assertLess(np.abs(fitted.x_validation).max(), 2.0)
        self.assertLessEqual(fitted.x_train[:, 0].max(), fitted.x_validation[0, 0])

    def test_target_vip_identifies_dominant_predictor(self) -> None:
        module = self.require_module()
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 5))
        y = np.column_stack(
            [4.0 * x[:, 0] + rng.normal(scale=0.1, size=120), -3.0 * x[:, 1]]
        )
        fitted, prediction = module.fit_pls(x[:90], y[:90], x[90:], n_components=2)
        vip = module.target_vip_scores(fitted)
        self.assertEqual(prediction.shape, (30, 2))
        self.assertEqual(vip.shape, (5, 2))
        self.assertEqual(int(np.argmax(vip[:, 0])), 0)
        self.assertEqual(int(np.argmax(vip[:, 1])), 1)

    def test_graph_penalty_pulls_signed_neighbor_coefficients_together(self) -> None:
        module = self.require_module()
        rng = np.random.default_rng(11)
        x1 = rng.normal(size=100)
        x2 = x1 + rng.normal(scale=0.08, size=100)
        x = np.column_stack([x1, x2])
        y = 3.0 * x1 + rng.normal(scale=0.15, size=100)
        laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
        no_graph = module.fit_graph_elasticnet(
            x, y, laplacian, lambda_l1=0.01, lambda_l2=0.01, lambda_graph=0.0
        )
        with_graph = module.fit_graph_elasticnet(
            x, y, laplacian, lambda_l1=0.01, lambda_l2=0.01, lambda_graph=10.0
        )
        self.assertLess(
            abs(with_graph.coef_[0] - with_graph.coef_[1]),
            abs(no_graph.coef_[0] - no_graph.coef_[1]),
        )

    def test_graph_penalty_respects_negative_edge_direction(self) -> None:
        module = self.require_module()
        rng = np.random.default_rng(13)
        x1 = rng.normal(size=120)
        x2 = -x1 + rng.normal(scale=0.08, size=120)
        x = np.column_stack([x1, x2])
        y = 2.0 * x1 - 2.0 * x2 + rng.normal(scale=0.1, size=120)
        signed_laplacian = np.array([[1.0, 1.0], [1.0, 1.0]])
        fitted = module.fit_graph_elasticnet(
            x,
            y,
            signed_laplacian,
            lambda_l1=0.005,
            lambda_l2=0.005,
            lambda_graph=5.0,
        )
        self.assertLess(fitted.coef_[0] * fitted.coef_[1], 0.0)
        self.assertLess(abs(fitted.coef_[0] + fitted.coef_[1]), 0.2)

    def test_multi_target_graph_elasticnet_matches_output_shape(self) -> None:
        module = self.require_module()
        rng = np.random.default_rng(17)
        x = rng.normal(size=(80, 4))
        y = np.column_stack([2.0 * x[:, 0], -3.0 * x[:, 1]])
        laplacian = np.zeros((4, 4))
        penalty = module.build_quadratic_penalty_rows(
            laplacian,
            n_samples=len(x),
            lambda_l2=0.1,
            lambda_graph=0.0,
        )
        fitted = module.fit_multi_target_with_penalty_rows(
            x, y, penalty, lambda_l1=0.01
        )
        self.assertEqual(fitted.coef_.shape, (2, 4))
        self.assertEqual(fitted.predict(x[:5]).shape, (5, 2))
        self.assertGreater(fitted.coef_[0, 0], 0.0)
        self.assertLess(fitted.coef_[1, 1], 0.0)

    def test_spearman_scores_recover_positive_and_negative_directions(self) -> None:
        module = self.require_module()
        x = pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": [4, 3, 2, 1, 0]})
        y = pd.DataFrame({"up": [0, 1, 2, 3, 4], "down": [4, 3, 2, 1, 0]})
        scores = module.spearman_target_scores(x, y)
        self.assertEqual(scores.loc["a", "up"], 1.0)
        self.assertEqual(scores.loc["a", "down"], -1.0)
        self.assertEqual(scores.loc["b", "up"], -1.0)

    def test_rank_table_filters_ineligible_candidates_and_breaks_ties(self) -> None:
        module = self.require_module()
        scores = pd.DataFrame(
            {"t1": [0.8, -0.9, 0.8]}, index=["b", "x", "a"]
        )
        eligible = pd.Series([True, False, True], index=scores.index)
        ranking = module.rank_score_matrix(scores, "demo", eligible)
        selected = ranking.query("target_code == 't1'")
        self.assertEqual(selected["feature_code"].tolist(), ["a", "b"])
        self.assertEqual(selected["rank"].tolist(), [1, 2])
        self.assertEqual(selected["direction"].tolist(), ["positive", "positive"])


if __name__ == "__main__":
    unittest.main()
