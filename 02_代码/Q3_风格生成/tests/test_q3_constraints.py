import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_constraints import audit_concentrations, project_upper_bounds, summarize_migration


class ConstraintTests(unittest.TestCase):
    def test_source_below_threshold_separates_scenarios(self):
        expected = {"C1": False, "C2": True, "C3": False}
        for scenario, passed in expected.items():
            result = audit_concentrations([1.], [1.5], [True], scenario, threshold=[2.])
            self.assertEqual(bool(result.passed.iloc[0]), passed)

    def test_source_above_threshold_separates_threshold_remediation(self):
        for scenario, passed in {"C1": True, "C2": True, "C3": False}.items():
            result = audit_concentrations([3.], [2.5], [True], scenario, threshold=[2.])
            self.assertEqual(bool(result.passed.iloc[0]), passed)

    def test_unknown_source_or_threshold_not_a_pass(self):
        for scenario in ["C1", "C2", "C3"]:
            result = audit_concentrations([1.], [.5], [False], scenario, threshold=[2.])
            self.assertTrue(pd.isna(result.passed.iloc[0]))
        for scenario in ["C2", "C3"]:
            result = audit_concentrations([1.], [.5], [True], scenario)
            self.assertEqual(result.status.iloc[0], "unknown_threshold")
            self.assertTrue(pd.isna(result.passed.iloc[0]))
        self.assertTrue(audit_concentrations([1.], [.5], [True], "C1").passed.iloc[0])

    def test_invalid_candidate_is_failure_even_if_source_unknown(self):
        result = audit_concentrations([np.nan]*3, [np.nan, np.inf, -1.], [False]*3, "C1")
        self.assertTrue(result.status.eq("invalid_candidate").all())
        self.assertTrue(result.passed.eq(False).all())

    def test_projection_never_raises_values_and_preserves_unknown_limit(self):
        x = np.array([.1, 3., 4.])
        original = x.copy()
        projected = project_upper_bounds(x, [2., 2., np.nan])
        np.testing.assert_array_equal(projected, [.1, 2., 4.])
        np.testing.assert_array_equal(x, original)

    def test_zero_migration_cannot_be_success(self):
        table = audit_concentrations([1.]*4, [1.]*4, [True]*4, "C1")
        result = summarize_migration(table, target_gain=False, proxy_nonworsening=True)
        self.assertEqual(result["status"], "no_target_gain")
        self.assertFalse(result["eligible"])

    def test_partial_certification_not_hidden_by_known_only_rate(self):
        table = audit_concentrations([1., 1.], [.5, .5], [True, False], "C1")
        result = summarize_migration(table, target_gain=True, proxy_nonworsening=True)
        self.assertEqual(result["decidable_pass_rate"], 1.)
        self.assertEqual(result["n_unknown"], 1)
        self.assertFalse(result["eligible"])

    def test_proxy_is_not_a_chemical_certificate(self):
        table = audit_concentrations([1.]*4, [.5]*4, [True]*4, "C1")
        for proxy, expected in [(None, "needs_proxy_verification"), (False, "rejected_proxy"),
                                (True, "eligible_statistical_candidate")]:
            result = summarize_migration(table, target_gain=True, proxy_nonworsening=proxy)
            self.assertEqual(result["status"], expected)
            self.assertFalse(result["physical_safety_certified"])

    def test_malformed_input_rejected(self):
        for args in [([1.], [1.], [True], "C9"), ([-1.], [1.], [True], "C1")]:
            with self.assertRaises(ValueError):
                audit_concentrations(*args)
        with self.assertRaises(ValueError):
            audit_concentrations([1.], [1.], [True], "C2", threshold=0.)


if __name__ == "__main__":
    unittest.main()
