from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_e203_shared_lowrank import validate_formal_config, verify_e202_gate


class E203RunnerTests(unittest.TestCase):
    def test_formal_config_freezes_cv_methods_and_training_only_selection(self):
        config = {
            "methods": ["shared_lowrank", "style_residual_shrunk", "style_residual_floor"],
            "target_styles": ["Tripel", "Kriek", "Lambic"],
            "seeds": [1, 2, 3, 4, 5],
            "n_repeats": 10,
            "n_folds": 3,
            "rank_candidates": [2, 4],
            "covariance_kappa_candidates": [10.0, 30.0],
            "inner_splits": 2,
            "floor_fields": {"X4VG": {"value": -2.0, "tolerance": 1e-9, "prior_strength": 10.0}},
            "gpu_used": False,
        }
        validate_formal_config(config)
        with self.assertRaises(ValueError):
            validate_formal_config(dict(config, methods=["style_residual_floor"]))
        with self.assertRaises(ValueError):
            validate_formal_config(dict(config, gpu_used=True))

    def test_gate_requires_complete_e202_and_matching_q1_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {
                "experiment_id": "E202",
                "status": "complete",
                "n_outer_folds": 30,
                "signature": {"q1_final_oof_sha256": "abc"},
            }
            (root / "q3_run_manifest_E202.json").write_text(json.dumps(manifest))
            result = verify_e202_gate(root, expected_q1_oof_sha256="abc")
            self.assertEqual(result["n_outer_folds"], 30)
            with self.assertRaises(ValueError):
                verify_e202_gate(root, expected_q1_oof_sha256="changed")


if __name__ == "__main__":
    unittest.main()
