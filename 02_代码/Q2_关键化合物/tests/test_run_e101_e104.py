from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

try:
    runner = importlib.import_module("run_e101_e104")
except ModuleNotFoundError:
    runner = None


class E101E104RunnerTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(runner, "run_e101_e104 module has not been implemented")
        return runner

    def test_select_best_candidate_breaks_ties_by_registration_order(self) -> None:
        module = self.require_module()
        candidates = [
            {"lambda_l1": 0.03, "lambda_l2": 0.1},
            {"lambda_l1": 0.01, "lambda_l2": 1.0},
        ]
        selected = module.select_best_candidate(candidates, [0.25, 0.25])
        self.assertEqual(selected, candidates[0])

    def test_artifact_suffix_is_safe_and_explicit(self) -> None:
        module = self.require_module()
        self.assertEqual(module.artifact_suffix("pilot_R1"), "_pilot_R1")
        self.assertEqual(module.artifact_suffix(""), "")
        with self.assertRaises(ValueError):
            module.artifact_suffix("bad label/with slash")


if __name__ == "__main__":
    unittest.main()
