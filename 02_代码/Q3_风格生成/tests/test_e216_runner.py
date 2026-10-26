from pathlib import Path
import sys
import unittest

import numpy as np
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_e216_box_migration import registered_bounds, validate_config


class E216RunnerTests(unittest.TestCase):
    def test_registered_bounds_do_not_invent_unknown_source_limits(self):
        lower, upper = registered_bounds([2.0, np.nan, 0.01, np.nan], [True, False, True, False],
                                         ["raw", "raw", "log10", "log10"])
        np.testing.assert_array_equal(lower, [0.0, 0.0, -np.inf, -np.inf])
        np.testing.assert_allclose(upper[[0, 2]], [2.0, -2.0])
        self.assertTrue(np.isinf(upper[[1, 3]]).all())

    def test_no_change_to_box_protocol_after_registration(self):
        config = yaml.safe_load((SRC.parent / "configs/E216_box_migration.yaml").read_text())
        validate_config(config)
        for key, value in [("mean_kappa", 20.0), ("direct_raw_lower_bound", -1.0),
                           ("path_lambdas", [0.5, 1.0]), ("auditor_used_during_search", True),
                           ("q1_retrained", True)]:
            with self.subTest(key=key):
                bad = dict(config, **{key: value})
                with self.assertRaises(ValueError):
                    validate_config(bad)


if __name__ == "__main__":
    unittest.main()
