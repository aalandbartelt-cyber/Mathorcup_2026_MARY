from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
stability = importlib.import_module("complementary_stability")


class ComplementaryStabilityTests(unittest.TestCase):
    def test_complementary_halves_are_disjoint_and_cover_rows(self) -> None:
        styles = np.repeat(["a", "b", "c"], [8, 6, 4])
        pairs = stability.stratified_complementary_pairs(styles, n_pairs=5, random_state=7)
        for left, right in pairs:
            self.assertEqual(set(left) & set(right), set())
            self.assertEqual(set(left) | set(right), set(range(len(styles))))

    def test_odd_strata_alternate_extra_row_and_repeat_is_deterministic(self) -> None:
        styles = np.repeat(["a", "b"], [5, 4])
        first = stability.stratified_complementary_pairs(styles, 4, random_state=11)
        second = stability.stratified_complementary_pairs(styles, 4, random_state=11)
        for (left_a, right_a), (left_b, right_b) in zip(first, second):
            np.testing.assert_array_equal(left_a, left_b)
            np.testing.assert_array_equal(right_a, right_b)
        self.assertNotEqual(len(first[0][0]), len(first[1][0]))

    def test_selection_probability_counts_both_halves(self) -> None:
        selections = [{0, 2}, {0}, {1}, {0, 1}]
        np.testing.assert_allclose(stability.selection_probability(selections, 3), [0.75, 0.5, 0.25])

    def test_empirical_fdp_is_bounded(self) -> None:
        self.assertEqual(stability.empirical_fdp(real_selected=8, null_selected=2), 0.25)
        self.assertEqual(stability.empirical_fdp(real_selected=0, null_selected=2), 1.0)

    def test_pair_jaccard_and_pfer(self) -> None:
        values = stability.complementary_pair_jaccard([{1, 2}, {1}], [{2, 3}, {1}])
        np.testing.assert_allclose(values, [1 / 3, 1.0])
        self.assertAlmostEqual(stability.mb_pfer_bound(10, 0.8, 218), 100 / (0.6 * 218))

    def test_confirmed_threshold_never_silently_relaxes_gate(self) -> None:
        result = stability.confirmed_threshold([0.6, 0.7], [0.6, 0.7], [0.6, 0.7])
        self.assertIsNone(result["threshold"])
        self.assertEqual(result["selected_indices"].tolist(), [])
        empty = stability.confirmed_threshold([0.1, 0.2], [0.1, 0.2], [0.8, 0.9])
        self.assertIsNone(empty["threshold"])


if __name__ == "__main__":
    unittest.main()
