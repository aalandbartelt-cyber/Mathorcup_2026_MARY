from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from q1_metrics import matrix_metrics, ordered_proxy_metrics  # noqa: E402


class Q1MetricTests(unittest.TestCase):
    def test_matrix_metrics_macro_average(self) -> None:
        y = np.column_stack([np.arange(14), np.arange(14)[::-1]]).astype(float)

        metrics = matrix_metrics(y, y.copy())

        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0)
        self.assertAlmostEqual(metrics["qwk_proxy"], 1.0)
        self.assertAlmostEqual(metrics["acc_pm1_proxy"], 1.0)

    def test_proxy_is_positive_affine_invariant(self) -> None:
        y = np.linspace(-2.0, 2.0, 70)

        self.assertEqual(
            ordered_proxy_metrics(y, y),
            ordered_proxy_metrics(y, 3.0 * y + 9.0),
        )

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            matrix_metrics(np.zeros((5, 2)), np.zeros((5, 3)))


if __name__ == "__main__":
    unittest.main()
