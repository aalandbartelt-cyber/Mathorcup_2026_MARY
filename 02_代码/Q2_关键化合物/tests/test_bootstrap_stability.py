from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

try:
    stability = importlib.import_module("bootstrap_stability")
except ModuleNotFoundError:
    stability = None


class BootstrapStabilityTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(stability, "bootstrap_stability module is missing")
        return stability

    def test_stratified_bootstrap_preserves_stratum_counts_and_oob(self) -> None:
        module = self.require_module()
        strata = np.array(["a"] * 5 + ["b"] * 7)
        sample, oob = module.stratified_bootstrap_indices(
            strata, np.random.default_rng(9)
        )
        self.assertEqual(len(sample), len(strata))
        self.assertEqual((strata[sample] == "a").sum(), 5)
        self.assertEqual((strata[sample] == "b").sum(), 7)
        self.assertEqual(len(np.intersect1d(np.unique(sample), oob)), 0)
        self.assertEqual(set(np.unique(sample)) | set(oob), set(range(len(strata))))

    def test_rank_scores_descending_breaks_ties_by_feature_code(self) -> None:
        module = self.require_module()
        scores = np.array([[0.5], [0.5], [0.1]])
        ranks = module.rank_scores_descending(scores, ["b", "a", "c"])
        self.assertEqual(ranks[:, 0].tolist(), [2, 1, 3])

    def test_consensus_ranks_prioritize_cross_method_top10(self) -> None:
        module = self.require_module()
        rank_a = np.array([[8], [1], [100]])
        rank_b = np.array([[8], [100], [1]])
        rank_c = np.array([[8], [100], [100]])
        ranks = module.consensus_ranks_from_methods(
            [rank_a, rank_b, rank_c], ["shared", "solo_a", "solo_b"]
        )
        self.assertEqual(ranks[:, 0].tolist(), [1, 2, 3])

    def test_update_coclustering_counts_pairs_in_same_module(self) -> None:
        module = self.require_module()
        counts = np.zeros((4, 4), dtype=np.uint16)
        module.update_coclustering(counts, [["a", "b"], ["c"], ["d"]], ["a", "b", "c", "d"])
        self.assertEqual(int(counts[0, 1]), 1)
        self.assertEqual(int(counts[1, 0]), 1)
        self.assertEqual(int(counts[0, 2]), 0)
        self.assertTrue(np.all(np.diag(counts) == 1))


if __name__ == "__main__":
    unittest.main()
