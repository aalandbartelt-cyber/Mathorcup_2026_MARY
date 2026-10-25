from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
gates = importlib.import_module("downstream_input_gates")


def dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class DownstreamInputGateTests(unittest.TestCase):
    def test_e118_rejects_missing_e117_without_generating_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)
            dump(result / "q2_run_manifest_E113.json", {"status": "complete", "n_outer_folds": 30, "n_features": 218})
            dump(result / "q2_run_manifest_E114.json", {"status": "complete", "n_outer_folds": 30})
            dump(
                result / "q2_run_manifest_E115_final_nnls.json",
                {
                    "status": "complete", "repeat_ids": list(range(10)),
                    "n_outer_folds": 30, "n_features_strict": 218, "n_targets": 50,
                },
            )
            audit = gates.audit_e118_inputs(result)
            self.assertEqual(audit["status"], "blocked")
            self.assertFalse(audit["formal_ranking_generated"])
            self.assertTrue(any("E117" in value for value in audit["problems"]))

    def test_e118_accepts_complete_contract_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)
            dump(result / "q2_run_manifest_E113.json", {"status": "complete", "n_outer_folds": 30, "n_features": 218})
            dump(result / "q2_run_manifest_E114.json", {"status": "complete", "n_outer_folds": 30})
            dump(
                result / "q2_run_manifest_E115_final_nnls.json",
                {
                    "status": "complete", "repeat_ids": list(range(10)),
                    "n_outer_folds": 30, "n_features_strict": 218, "n_targets": 50,
                },
            )
            dump(
                result / "q2_run_manifest_E117.json",
                {"status": "complete", "formal_run": True, "n_features_strict": 218, "n_targets": 50},
            )
            audit = gates.audit_e118_inputs(result)
            self.assertEqual(audit["status"], "ready")
            self.assertFalse(audit["formal_ranking_generated"])

    def test_e119_requires_promoted_unchanged_150_row_statistical_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)
            dump(result / "q2_run_manifest_E118.json", {"status": "complete"})
            dump(result / "q2_replacement_decision_E118.json", {"promoted": True})
            table = pd.DataFrame(
                [
                    {"target_code": f"t{target}", "consensus_rank": rank, "feature_code": f"x{rank}"}
                    for target in range(50) for rank in (1, 2, 3)
                ]
            )
            table.to_csv(result / "q2_final_top3_statistical_E118.csv", index=False)
            audit = gates.audit_e119_inputs(result)
            self.assertEqual(audit["status"], "ready")
            self.assertFalse(audit["chemical_grading_generated"])
            self.assertFalse(audit["reranking_permitted"])


if __name__ == "__main__":
    unittest.main()
