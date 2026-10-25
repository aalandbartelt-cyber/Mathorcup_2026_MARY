from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ordered_models import (  # noqa: E402
    decode_expected_level,
    expected_level_to_continuous,
    fit_predict_cumulative_hgb,
    project_survival_probabilities,
    training_quantile_anchors,
)


class OrderedModelTests(unittest.TestCase):
    def test_training_anchors_equal_training_quantiles(self) -> None:
        y_train = np.column_stack(
            [np.arange(14, dtype=float), np.arange(14, dtype=float) * 2]
        )

        anchors = training_quantile_anchors(y_train)

        expected = np.quantile(y_train, np.linspace(0.0, 1.0, 7), axis=0).T
        np.testing.assert_allclose(anchors, expected)

    def test_projection_produces_nonincreasing_survival(self) -> None:
        raw = np.array([[[0.2, 0.9, 0.4, 1.2, -0.1, 0.3]]])

        projected = project_survival_probabilities(raw)

        self.assertTrue(((projected >= 0.0) & (projected <= 1.0)).all())
        self.assertTrue((np.diff(projected, axis=-1) <= 1e-12).all())

    def test_expected_level_and_continuous_decode_are_monotone(self) -> None:
        survival = np.array(
            [
                [[1, 1, 1, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 0]],
            ],
            dtype=float,
        )
        anchors = np.arange(7, dtype=float).reshape(1, -1)

        decoded = expected_level_to_continuous(
            decode_expected_level(survival), anchors
        )

        self.assertLess(decoded[0, 0], decoded[1, 0])

    def test_cumulative_hgb_is_finite_and_deterministic_with_constant_target(
        self,
    ) -> None:
        x = np.arange(48, dtype=float).reshape(16, 3)
        x[2, 1] = np.nan
        y = np.column_stack([np.zeros(16), np.linspace(-1, 1, 16)])
        kwargs = {
            "max_leaf_nodes": 7,
            "min_samples_leaf": 4,
            "learning_rate": 0.05,
            "l2_regularization": 1.0,
            "max_iter": 20,
            "random_state": 17,
        }

        first = fit_predict_cumulative_hgb(x[:12], y[:12], x[12:], **kwargs)
        second = fit_predict_cumulative_hgb(x[:12], y[:12], x[12:], **kwargs)

        self.assertEqual(first.continuous.shape, (4, 2))
        self.assertEqual(first.survival_probability.shape, (4, 2, 6))
        self.assertTrue(np.isfinite(first.continuous).all())
        np.testing.assert_allclose(first.continuous, second.continuous)
        np.testing.assert_allclose(first.continuous[:, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
