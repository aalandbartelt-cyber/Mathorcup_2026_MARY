from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

try:
    robust_consensus = importlib.import_module("robust_consensus")
except ModuleNotFoundError:
    robust_consensus = None


class RobustConsensusTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(robust_consensus, "robust_consensus module is missing")
        return robust_consensus

    def test_leave_one_style_aggregation_prioritizes_top3_survival(self):
        module = self.require_module()
        frame = pd.DataFrame(
            {
                "omitted_style": ["s1"] * 4 + ["s2"] * 4,
                "target_code": ["y"] * 8,
                "feature_code": ["a", "b", "c", "d"] * 2,
                "consensus_rank": [1, 2, 3, 4, 1, 4, 2, 3],
            }
        )
        result = module.aggregate_leave_one_style_ranks(frame, expected_styles=2)
        self.assertEqual(result.feature_code.tolist(), ["a", "c", "b", "d"])
        self.assertEqual(result.aggregate_rank.tolist(), [1, 2, 3, 4])
        frequencies = result.set_index("feature_code")["top3_survival_frequency"]
        self.assertEqual(float(frequencies["a"]), 1.0)
        self.assertEqual(float(frequencies["b"]), 0.5)

    def test_leave_one_style_requires_complete_style_coverage(self):
        module = self.require_module()
        frame = pd.DataFrame(
            {
                "omitted_style": ["s1", "s1"],
                "target_code": ["y", "y"],
                "feature_code": ["a", "b"],
                "consensus_rank": [1, 2],
            }
        )
        with self.assertRaises(ValueError):
            module.aggregate_leave_one_style_ranks(frame, expected_styles=2)


if __name__ == "__main__":
    unittest.main()
