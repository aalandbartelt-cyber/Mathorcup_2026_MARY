from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from hierarchical_importance import (  # noqa: E402
    delta_mae,
    fit_conditional_feature_model,
    joint_permutation,
    residual_permutation,
)


class HierarchicalImportanceTests(unittest.TestCase):
    def test_joint_permutation_keeps_module_columns_together(self) -> None:
        x = np.arange(60, dtype=float).reshape(12, 5)

        changed, order = joint_permutation(x, [1, 3], np.random.default_rng(4))

        np.testing.assert_array_equal(changed[:, 1], x[order, 1])
        np.testing.assert_array_equal(changed[:, 3], x[order, 3])
        np.testing.assert_array_equal(changed[:, 0], x[:, 0])

    def test_residual_permutation_preserves_conditional_mean(self) -> None:
        mean = np.linspace(0, 1, 20)
        residual = np.arange(20, dtype=float)

        changed = residual_permutation(mean, residual, np.random.default_rng(5))

        np.testing.assert_allclose(np.sort(changed - mean), np.sort(residual))

    def test_conditional_feature_model_handles_unseen_style(self) -> None:
        rng = np.random.default_rng(7)
        conditioners = rng.normal(size=(60, 2))
        style = np.repeat(["A", "B"], 30)
        feature = conditioners[:, 0] + (style == "B") + rng.normal(scale=0.1, size=60)
        model = fit_conditional_feature_model(conditioners, feature, style)

        prediction = model.predict(conditioners[:3], np.array(["C", "C", "C"]))

        self.assertEqual(prediction.shape, (3,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_delta_mae_is_positive_when_permutation_hurts(self) -> None:
        truth = np.arange(5, dtype=float)
        baseline = truth.copy()
        changed = truth[::-1].copy()

        self.assertGreater(delta_mae(truth, baseline, changed), 0.0)


if __name__ == "__main__":
    unittest.main()
