from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from compute_q1_extended_metrics import (  # noqa: E402
    ordered_proxy_metrics,
    quantile_levels,
)


class OrderedMetricTests(unittest.TestCase):
    def test_separate_quantile_calibration_preserves_perfect_order(self) -> None:
        y_true = np.linspace(-3.0, 3.0, 70)
        y_pred = 4.0 * y_true + 11.0

        metrics = ordered_proxy_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["qwk_proxy"], 1.0)
        self.assertAlmostEqual(metrics["acc_proxy"], 1.0)
        self.assertAlmostEqual(metrics["acc_pm1_proxy"], 1.0)
        self.assertAlmostEqual(metrics["level_mae_proxy"], 0.0)

    def test_reversed_order_is_penalized(self) -> None:
        y_true = np.linspace(-3.0, 3.0, 70)
        y_pred = -y_true

        metrics = ordered_proxy_metrics(y_true, y_pred)

        self.assertLess(metrics["qwk_proxy"], 0.0)
        self.assertLess(metrics["acc_pm1_proxy"], 0.5)

    def test_quantile_levels_stays_in_zero_to_six_with_ties(self) -> None:
        reference = np.array([0.0] * 30 + [1.0] * 20 + [2.0] * 20)
        levels = quantile_levels(reference, reference)

        self.assertGreaterEqual(levels.min(), 0)
        self.assertLessEqual(levels.max(), 6)
        self.assertTrue(np.issubdtype(levels.dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
