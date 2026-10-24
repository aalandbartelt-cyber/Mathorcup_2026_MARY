from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ensemble_models import (  # noqa: E402
    apply_convex_blend,
    fit_predict_hgb_multioutput,
    select_feasible_convex_blend,
    shrink_target_weights,
    simplex_weight_grid,
)


class EnsembleModelTests(unittest.TestCase):
    def test_simplex_grid_is_nonnegative_unique_and_sums_to_one(self) -> None:
        weights = simplex_weight_grid(n_models=3, step=0.5)

        self.assertEqual(len(weights), 6)
        self.assertEqual(len(np.unique(weights, axis=0)), len(weights))
        self.assertTrue((weights >= 0).all())
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)

    def test_apply_convex_blend_matches_manual_average(self) -> None:
        first = np.arange(12, dtype=float).reshape(6, 2)
        second = first + 2.0
        stack = np.stack([first, second], axis=-1)

        blended = apply_convex_blend(stack, np.array([0.25, 0.75]))

        np.testing.assert_allclose(blended, 0.25 * first + 0.75 * second)

    def test_feasible_blend_rejects_continuous_regression(self) -> None:
        y = np.arange(20, dtype=float).reshape(10, 2)
        perfect = y.copy()
        reversed_prediction = y[::-1].copy()
        stack = np.stack([perfect, reversed_prediction], axis=-1)

        result = select_feasible_convex_blend(
            y,
            stack,
            simplex_weight_grid(2, 0.5),
            r2_floor=0.9,
            mae_ceiling=0.1,
        )

        self.assertTrue(result.feasible)
        np.testing.assert_allclose(result.weights, [1.0, 0.0])

    def test_target_weights_shrink_to_global_weight(self) -> None:
        target = np.array([[1.0, 0.0], [0.0, 1.0]])
        global_weight = np.array([0.75, 0.25])

        shrunk = shrink_target_weights(target, global_weight, shrinkage=1.0)

        np.testing.assert_allclose(shrunk, np.tile(global_weight, (2, 1)))

    def test_hgb_multioutput_is_finite_and_deterministic(self) -> None:
        x = np.arange(60, dtype=float).reshape(20, 3)
        x[3, 1] = np.nan
        y = np.column_stack([x[:, 0], np.nan_to_num(x[:, 1], nan=0.0)])
        kwargs = {
            "max_leaf_nodes": 7,
            "min_samples_leaf": 5,
            "learning_rate": 0.05,
            "l2_regularization": 1.0,
            "max_iter": 20,
            "random_state": 9,
        }

        first = fit_predict_hgb_multioutput(x[:15], y[:15], x[15:], **kwargs)
        second = fit_predict_hgb_multioutput(x[:15], y[:15], x[15:], **kwargs)

        self.assertEqual(first.shape, (5, 2))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
