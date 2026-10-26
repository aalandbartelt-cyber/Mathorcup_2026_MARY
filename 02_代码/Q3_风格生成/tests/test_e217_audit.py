from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_e217_local_pair_search import verify_committed_lock, decide_exploratory


class E217AuditTests(unittest.TestCase):
    def test_lock_must_be_committed_and_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for command in (["git", "init", "-q"], ["git", "config", "user.name", "Test"],
                            ["git", "config", "user.email", "test@example.com"]):
                subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)
            output = root / "selected.csv"
            output.write_text("beer,target\n230,Lambic\n")
            lock = root / "lock.json"
            content = {"status": "locked_before_model_family_audit", "auditor_accessed": False,
                       "n_source_target_pairs": 33, "n_search_points": 63756,
                       "output_sha256": {output.name: hashlib.sha256(output.read_bytes()).hexdigest()}}
            lock.write_text(json.dumps(content) + "\n")
            subprocess.run(["git", "add", "lock.json", "selected.csv"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "lock"], cwd=root, check=True)
            self.assertEqual(verify_committed_lock(root, lock)["n_source_target_pairs"], 33)
            lock.write_text(lock.read_text() + " ")
            with self.assertRaises(ValueError):
                verify_committed_lock(root, lock)

    def test_only_fully_observed_lambic_candidate_with_both_margins_counts(self):
        candidate = dict(beer=230, target_style="Lambic", candidate_available=True,
                         n_direct_failed=0, n_direct_unknown=0, support_repeat_fraction=1.,
                         relative_target_gain=.2, primary_mean_delta=-.03,
                         primary_repeat_q90_delta=-.02, audit_mean_delta=-.02,
                         audit_repeat_q90_delta=-.011)
        self.assertTrue(decide_exploratory(candidate))
        self.assertFalse(decide_exploratory(dict(candidate, n_direct_unknown=1)))
        self.assertFalse(decide_exploratory(dict(candidate, audit_repeat_q90_delta=-.005)))
        self.assertFalse(decide_exploratory(dict(candidate, target_style="Kriek")))


if __name__ == "__main__":
    unittest.main()
