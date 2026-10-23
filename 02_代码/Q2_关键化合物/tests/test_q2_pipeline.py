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
    pipeline = importlib.import_module("q2_pipeline")
except ModuleNotFoundError:
    pipeline = None


class Q2PipelineTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(pipeline, "q2_pipeline module has not been implemented")
        return pipeline

    def test_frozen_outer_folds_cover_every_row_once_per_repeat(self) -> None:
        module = self.require_module()
        folds = module.load_frozen_outer_folds(n_samples=250, repeats=1)
        self.assertEqual(len(folds), 3)
        validation = np.concatenate([fold.validation_index for fold in folds])
        self.assertEqual(sorted(validation.tolist()), list(range(250)))
        for fold in folds:
            self.assertEqual(len(np.intersect1d(fold.train_index, fold.validation_index)), 0)
            self.assertEqual(len(fold.train_index) + len(fold.validation_index), 250)

    def test_extend_laplacian_leaves_missing_indicators_unpenalized(self) -> None:
        module = self.require_module()
        base = np.array([[1.0, -1.0], [-1.0, 1.0]])
        extended = module.extend_laplacian(base, total_features=4)
        np.testing.assert_allclose(extended[:2, :2], base)
        np.testing.assert_allclose(extended[2:, :], 0.0)
        np.testing.assert_allclose(extended[:, 2:], 0.0)

    def test_aggregate_fold_rankings_uses_median_rank_then_feature_code(self) -> None:
        module = self.require_module()
        rankings = pd.DataFrame(
            {
                "method": ["m"] * 6,
                "target_code": ["t"] * 6,
                "feature_code": ["b", "a", "c", "b", "a", "c"],
                "rank": [1, 2, 3, 2, 1, 3],
                "score": [0.8, 0.7, -0.1, 0.6, 0.9, -0.2],
                "split_id": ["f1"] * 3 + ["f2"] * 3,
            }
        )
        aggregate = module.aggregate_fold_rankings(rankings)
        self.assertEqual(aggregate["feature_code"].tolist(), ["a", "b", "c"])
        self.assertEqual(aggregate["aggregate_rank"].tolist(), [1, 2, 3])
        self.assertEqual(aggregate.iloc[-1]["direction"], "negative")

    def test_sparse_ranking_never_places_always_zero_ahead_of_selected(self) -> None:
        module = self.require_module()
        rankings = pd.DataFrame(
            {
                "method": ["sparse"] * 6,
                "target_code": ["t"] * 6,
                "feature_code": ["a", "z", "b", "a", "z", "b"],
                "rank": [2, 1, 3, 2, 1, 3],
                "score": [0.0, 0.3, 0.0, 0.0, 0.2, -0.1],
                "split_id": ["f1"] * 3 + ["f2"] * 3,
            }
        )
        aggregate = module.aggregate_fold_rankings(rankings)
        self.assertEqual(aggregate["feature_code"].tolist(), ["z", "b", "a"])
        self.assertEqual(aggregate["selection_frequency"].tolist(), [1.0, 0.5, 0.0])
        self.assertEqual(aggregate.iloc[0]["direction"], "positive")
        self.assertEqual(aggregate.iloc[1]["direction"], "negative")

    def test_metrics_keep_negative_r2_and_report_spearman(self) -> None:
        module = self.require_module()
        y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_pred = np.array([[3.0], [2.0], [1.0], [0.0]])
        metrics = module.metrics_by_target(y_true, y_pred, ["t"], "m")
        self.assertLess(metrics.iloc[0]["r2"], 0.0)
        self.assertEqual(metrics.iloc[0]["spearman"], -1.0)


if __name__ == "__main__":
    unittest.main()
