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
    importance = importlib.import_module("permutation_importance")
except ModuleNotFoundError:
    importance = None


class PermutationImportanceTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(
            importance, "permutation_importance module has not been implemented"
        )
        return importance

    def test_training_quantile_groups_do_not_fit_validation_values(self) -> None:
        module = self.require_module()
        training = np.arange(20, dtype=float)
        edges = module.fit_quantile_edges(training, n_bins=4)
        first = module.assign_quantile_groups(np.array([-999.0, 5.0, 999.0]), edges)
        second = module.assign_quantile_groups(np.array([-1e9, 5.0, 1e9]), edges)
        self.assertEqual(first[1], second[1])
        np.testing.assert_allclose(edges[[0, -1]], [-np.inf, np.inf])

    def test_grouped_row_permutation_preserves_groups_and_joint_rows(self) -> None:
        module = self.require_module()
        values = np.array([[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]])
        groups = np.array([0, 0, 0, 1, 1])
        order = module.grouped_permutation_order(
            groups, np.random.default_rng(17)
        )
        permuted = values[order]
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            self.assertEqual(set(order[rows].tolist()), set(rows.tolist()))
        np.testing.assert_array_equal(permuted[:, 1] - permuted[:, 0], 10)

    def test_connected_modules_use_only_eligible_features(self) -> None:
        module = self.require_module()
        corr = pd.DataFrame(
            [[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        modules = module.connected_modules(
            corr, eligible_features=["a", "b"], threshold=0.7
        )
        self.assertEqual(modules, [["a", "b"]])

    def test_strongest_neighbor_uses_absolute_training_correlation(self) -> None:
        module = self.require_module()
        corr = pd.DataFrame(
            [[1.0, -0.9, 0.8], [-0.9, 1.0, 0.1], [0.8, 0.1, 1.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        self.assertEqual(module.strongest_neighbor(corr, "a"), "b")

    def test_importance_aggregation_does_not_reward_harmful_negative_delta(self) -> None:
        module = self.require_module()
        rows = pd.DataFrame(
            {
                "method": ["m"] * 6,
                "target_code": ["t"] * 6,
                "feature_code": ["good", "weak", "harm", "good", "weak", "harm"],
                "split_id": ["f1"] * 3 + ["f2"] * 3,
                "rank": [1, 2, 3, 1, 2, 3],
                "score": [0.3, 0.1, -5.0, 0.2, 0.05, -4.0],
                "selected_top3": [True, True, True, False, True, True],
            }
        )
        aggregate = module.aggregate_feature_importance(rows)
        self.assertEqual(set(aggregate.feature_code.iloc[:2]), {"good", "weak"})
        self.assertEqual(aggregate.feature_code.iloc[-1], "harm")
        self.assertLess(aggregate.iloc[-1].median_score, 0.0)


if __name__ == "__main__":
    unittest.main()
