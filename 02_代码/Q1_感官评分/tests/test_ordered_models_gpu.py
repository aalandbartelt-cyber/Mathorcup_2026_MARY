from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ordered_models_gpu import fit_predict_cumulative_xgb  # noqa: E402


class OrderedGPUModelTests(unittest.TestCase):
    def test_cuda_cumulative_xgb_is_finite_and_deterministic(self) -> None:
        x = np.arange(48, dtype=float).reshape(16, 3)
        x[2, 1] = np.nan
        y = np.column_stack([np.zeros(16), np.linspace(-1, 1, 16)])
        kwargs = {
            "max_depth": 3,
            "min_child_weight": 2.0,
            "learning_rate": 0.05,
            "n_estimators": 12,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 17,
        }

        first = fit_predict_cumulative_xgb(x[:12], y[:12], x[12:], **kwargs)
        second = fit_predict_cumulative_xgb(x[:12], y[:12], x[12:], **kwargs)

        self.assertEqual(first.continuous.shape, (4, 2))
        self.assertTrue(np.isfinite(first.continuous).all())
        np.testing.assert_allclose(first.continuous, second.continuous, atol=1e-8)
        np.testing.assert_allclose(first.continuous[:, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
