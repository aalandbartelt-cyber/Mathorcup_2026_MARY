from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

try:
    consensus = importlib.import_module("rank_consensus")
except ModuleNotFoundError:
    consensus = None


class RankConsensusTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(consensus, "rank_consensus module is missing")
        return consensus

    def test_rrf_prioritizes_cross_method_top10_support(self) -> None:
        module = self.require_module()
        table = pd.DataFrame(
            {
                "method": ["a", "a", "b", "b", "c", "c"],
                "target_code": ["t"] * 6,
                "feature_code": ["shared", "solo", "shared", "solo", "shared", "solo"],
                "aggregate_rank": [8, 1, 8, 100, 8, 100],
            }
        )
        result = module.reciprocal_rank_consensus(table, expected_methods=3)
        self.assertEqual(result.iloc[0].feature_code, "shared")
        self.assertEqual(result.iloc[0].method_top10_support, 3)

    def test_rrf_requires_complete_method_coverage(self) -> None:
        module = self.require_module()
        incomplete = pd.DataFrame(
            {
                "method": ["a", "b"],
                "target_code": ["t", "t"],
                "feature_code": ["x", "x"],
                "aggregate_rank": [1, 1],
            }
        )
        with self.assertRaises(ValueError):
            module.reciprocal_rank_consensus(incomplete, expected_methods=3)


if __name__ == "__main__":
    unittest.main()
