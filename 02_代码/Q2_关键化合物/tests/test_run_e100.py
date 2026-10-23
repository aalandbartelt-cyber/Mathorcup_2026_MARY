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

try:
    runner = importlib.import_module("run_e100_contract")
except ModuleNotFoundError:
    runner = None


class E100RunnerTests(unittest.TestCase):
    def test_e100_writes_complete_machine_readable_contract(self) -> None:
        self.assertIsNotNone(runner, "run_e100_contract module has not been implemented")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runner.run_e100(output)
            expected = {
                "q2_candidate_dictionary.csv",
                "q2_graph_nodes.csv",
                "q2_graph_edges.csv.gz",
                "q2_graph_threshold_audit.csv",
                "q2_target_dictionary.csv",
                "q2_data_contract.json",
                "q2_run_manifest_E100.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)

            candidates = pd.read_csv(output / "q2_candidate_dictionary.csv")
            edges = pd.read_csv(output / "q2_graph_edges.csv.gz")
            thresholds = pd.read_csv(output / "q2_graph_threshold_audit.csv")
            targets = pd.read_csv(output / "q2_target_dictionary.csv")
            contract = json.loads((output / "q2_data_contract.json").read_text())
            manifest = json.loads((output / "q2_run_manifest_E100.json").read_text())

            self.assertEqual(len(candidates), 231)
            self.assertEqual(int(candidates["eligible_strict"].sum()), 218)
            self.assertEqual(len(edges), 231 * 230 // 2)
            self.assertEqual(len(targets), 50)
            object_values = targets.select_dtypes(include="object").astype(str)
            self.assertFalse(
                object_values.apply(
                    lambda column: column.str.contains(r"[\r\n]", regex=True).any()
                ).any(),
                "exported target dictionary must not contain hidden CR/LF characters",
            )
            row_05 = thresholds.loc[
                thresholds["abs_rho_threshold"].round(1).eq(0.5)
            ].iloc[0]
            self.assertEqual(int(row_05["n_edges"]), 645)
            self.assertEqual(int(row_05["isolated_nodes"]), 48)
            self.assertEqual(contract["graph_definition"], "spearman")
            self.assertLess(contract["official_graph_audit"]["spearman_max_abs_diff"], 1e-12)
            self.assertEqual(contract["candidate_scope"]["strict_count"], 218)
            self.assertEqual(manifest["experiment_id"], "E100")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["supervised_model_fitted"], False)


if __name__ == "__main__":
    unittest.main()
