from pathlib import Path
import json
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_e204_constrained_migration import validate_formal_config, verify_e204_gate


class E204RunnerTests(unittest.TestCase):
    def test_config_freezes_scenarios_model_and_nonzero_path(self):
        config = {
            "source_style": "Pils/Lager",
            "target_styles": ["Tripel", "Kriek", "Lambic"],
            "distribution_model": "style_copula_shrunk",
            "generated_pool_per_style": 2000,
            "scenarios": ["C1", "C2", "C3"],
            "path_lambdas": [0.25, 0.5, 0.75, 1.0],
            "minimum_relative_target_gain": 0.01,
            "stale_hops_proxy": "deferred_to_E205",
            "gpu_used": False,
        }
        validate_formal_config(config)
        with self.assertRaises(ValueError):
            validate_formal_config(dict(config, path_lambdas=[0.0, 1.0]))
        with self.assertRaises(ValueError):
            validate_formal_config(dict(config, distribution_model="shared_lowrank"))

    def test_gate_requires_audited_e203_copula_decision_and_q1_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "q3_run_manifest_E203.json").write_text(
                json.dumps({
                    "experiment_id": "E203",
                    "status": "complete",
                    "n_outer_folds": 30,
                    "signature": {"q1_final_oof_sha256": "abc"},
                })
            )
            (root / "q3_e203_audit_manifest.json").write_text(json.dumps({"status": "complete"}))
            (root / "q3_e203_replacement_decision.json").write_text(
                json.dumps({
                    "replace_copula_baseline": False,
                    "e204_primary_distribution_model": "style_copula_shrunk",
                    "e204_gate": "open_with_copula_primary_and_lowrank_ablation",
                })
            )
            gate = verify_e204_gate(root, expected_q1_oof_sha256="abc")
            self.assertEqual(gate["distribution_model"], "style_copula_shrunk")
            with self.assertRaises(ValueError):
                verify_e204_gate(root, expected_q1_oof_sha256="changed")


if __name__ == "__main__":
    unittest.main()
