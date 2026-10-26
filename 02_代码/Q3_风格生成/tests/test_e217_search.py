from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_e217_local_pair_search import select_primary, validate_ledger, validate_config


class E217SearchTests(unittest.TestCase):
    def test_support_is_hard_gate_and_minimum_change_selects_stably(self):
        rows = pd.DataFrame({
            "numeric_valid": [True] * 4,
            "relative_target_gain": [.02, .03, .04, .05],
            "source_change_rms": [.10, .20, .20, .50],
            "support_repeat_fraction": [.8, 1., 1., 1.],
            "primary_mean_delta": [-.04, -.02, -.03, -.05],
            "primary_repeat_q90_delta": [-.03, -.02, -.02, -.04],
            "pair_index": [0, 1, 2, 3], "lambda": [.1, .2, .2, .5],
        })
        index, feasible = select_primary(rows)
        np.testing.assert_array_equal(feasible, [False, True, True, True])
        self.assertEqual(index, 2)  # equal change: larger target gain wins
        rows.loc[:, "support_repeat_fraction"] = .8
        index, feasible = select_primary(rows)
        self.assertIsNone(index)
        self.assertFalse(feasible.any())

    def test_skipped_proxy_nan_cannot_become_feasible(self):
        rows = pd.DataFrame({
            "numeric_valid": [True, False], "relative_target_gain": [.02, .4],
            "source_change_rms": [.1, .05], "support_repeat_fraction": [1., 1.],
            "primary_mean_delta": [np.nan, -.2], "primary_repeat_q90_delta": [np.nan, -.2],
            "pair_index": [0, 1], "lambda": [.1, .2],
        })
        index, feasible = select_primary(rows)
        self.assertIsNone(index)
        self.assertFalse(feasible.any())

    def test_complete_ledger_rejects_missing_and_duplicate_paths(self):
        rows = pd.DataFrame([
            dict(beer=1, target_style=style, pair_index=pair, path_index=index,
                 **{"lambda": lam})
            for style in ("Tripel", "Kriek") for pair in range(2)
            for index, lam in enumerate((.1, .5))
        ])
        self.assertTrue(validate_ledger(rows, n_sources=1, target_styles=["Tripel", "Kriek"],
                                        pairs_per_group=2, lambdas=[.1, .5]))
        with self.assertRaises(ValueError):
            validate_ledger(rows.iloc[:-1], n_sources=1, target_styles=["Tripel", "Kriek"],
                            pairs_per_group=2, lambdas=[.1, .5])
        with self.assertRaises(ValueError):
            validate_ledger(pd.concat([rows, rows.iloc[[0]]]), n_sources=1,
                            target_styles=["Tripel", "Kriek"], pairs_per_group=2,
                            lambdas=[.1, .5])

    def test_registered_protocol_rejects_posthoc_changes(self):
        config = yaml.safe_load((SRC.parent / "configs/E217_local_pair_migration.yaml").read_text())
        self.assertTrue(validate_config(config))
        for key, value in [("nearest_anchors", 32), ("pair_weight", .75),
                           ("auditor_used_during_search", True), ("q1_retrained", True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(dict(config, **{key: value}))


if __name__ == "__main__":
    unittest.main()
