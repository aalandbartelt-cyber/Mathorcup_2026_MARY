from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_e216_sensory_lock import choose_locked_path, validate_search_geometry


class E216SensoryLockTests(unittest.TestCase):
    def test_geometry_requires_all_registered_pairs_and_paths(self):
        rows = []
        for beer in range(11):
            for style in ("Tripel", "Kriek", "Lambic"):
                for index, lam in enumerate((0, .05, .1, .2, .35, .5, .75, 1.)):
                    rows.append(dict(beer=beer, target_style=style, path_index=index,
                                     **{"lambda": lam}, n_direct_failed=0))
        paths = pd.DataFrame(rows)
        self.assertTrue(validate_search_geometry(paths))
        with self.assertRaises(ValueError):
            validate_search_geometry(paths.iloc[:-1])
        broken = paths.copy()
        broken.loc[1, "n_direct_failed"] = 1
        with self.assertRaises(ValueError):
            validate_search_geometry(broken)

    def test_lock_uses_minimum_change_among_primary_feasible_only(self):
        group = pd.DataFrame({
            "path_index": [1, 2, 3], "relative_target_gain": [.03, .08, .11],
            "source_change_rms": [.2, .3, .4], "primary_mean_delta": [-.02, -.04, -.05],
            "primary_repeat_q90_delta": [-.02, -.02, -.03],
            "support_repeat_fraction": [.8, 1., 1.],
        })
        index, feasible = choose_locked_path(group)
        np.testing.assert_array_equal(feasible, [False, True, True])
        self.assertEqual(index, 1)
        group.loc[:, "primary_repeat_q90_delta"] = 0
        index, feasible = choose_locked_path(group)
        self.assertIsNone(index)
        self.assertFalse(feasible.any())


if __name__ == "__main__":
    unittest.main()
