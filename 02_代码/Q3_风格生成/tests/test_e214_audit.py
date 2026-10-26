from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from audit_e214 import validate_manifest_structure


class E214AuditTests(unittest.TestCase):
    def test_manifest_requires_complete_cpu_candidate(self):
        base = {
            "experiment_id": "E214", "status": "complete", "n_outer_folds": 30,
            "q1_retrained": False, "nnls_refit": False, "gpu_used": False,
        }
        self.assertTrue(validate_manifest_structure(base))
        for key, value in [("status", "running"), ("n_outer_folds", 1), ("gpu_used", True), ("nnls_refit", True)]:
            with self.subTest(key=key):
                bad = dict(base, **{key: value})
                with self.assertRaises(ValueError):
                    validate_manifest_structure(bad)


if __name__ == "__main__":
    unittest.main()
