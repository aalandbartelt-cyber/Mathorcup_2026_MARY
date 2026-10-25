from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from stacking_diagnostics import pairwise_complementarity, validate_oof_alignment
from q1_metrics import ordered_proxy_metrics


class StackingDiagnosticTests(unittest.TestCase):
    def test_identical_predictions_have_no_exclusive_rescue(self):
        y = np.arange(21, dtype=float)[:, None]
        prediction = np.roll(y, 7, axis=0)[None]
        rows = pairwise_complementarity(y, {"a": prediction, "b": prediction}, ["t"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["first_rescued_by_second"], 0.)
        self.assertEqual(rows.iloc[0]["second_rescued_by_first"], 0.)
        self.assertAlmostEqual(rows.iloc[0]["residual_correlation"], 1.)

    def test_complementary_errors_rescue_each_other(self):
        y = np.arange(28, dtype=float)[:, None]
        a, b = y.copy(), y.copy()
        a[:14] = a[:14][::-1]
        b[14:] = b[14:][::-1]
        row = pairwise_complementarity(y, {"a": a[None], "b": b[None]}, ["t"]).iloc[0]
        self.assertGreater(row["first_rescued_by_second"], 0.)
        self.assertGreater(row["second_rescued_by_first"], 0.)
        self.assertAlmostEqual(row["oracle_union_pm1_diagnostic"], 1.)

    def test_positive_affine_map_does_not_improve_proxy_metrics(self):
        rng = np.random.default_rng(19)
        y, prediction = rng.normal(size=80), rng.normal(size=80)
        before = ordered_proxy_metrics(y, prediction)
        after = ordered_proxy_metrics(y, prediction * 3.2 - 4.)
        for name in before:
            self.assertAlmostEqual(before[name], after[name], places=12)

    def test_archive_alignment_rejects_reordered_beer_or_targets(self):
        a = {"beer": np.arange(6), "target_code": np.array(["a", "b"]),
             "repeat": np.array([0]), "style": np.array(["s"] * 6),
             "fold_assignment": np.array([[0, 0, 1, 1, 2, 2]]),
             "y_true": np.zeros((6, 2))}
        validate_oof_alignment(a, a)
        for field in ["beer", "target_code", "fold_assignment", "y_true"]:
            b = {k: v.copy() for k, v in a.items()}
            b[field] = b[field][::-1].copy()
            if field in ["fold_assignment", "y_true"]:
                b[field].flat[0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_oof_alignment(a, b)

    def test_incomplete_prediction_is_not_a_diagnostic_candidate(self):
        with self.assertRaises(ValueError):
            pairwise_complementarity(np.ones((5, 1)),
                                    {"a": np.full((1, 5, 1), np.nan),
                                     "b": np.ones((1, 5, 1))}, ["t"])


if __name__ == "__main__":
    unittest.main()
