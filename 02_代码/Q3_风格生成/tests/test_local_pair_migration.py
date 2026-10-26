from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from local_pair_migration import nearest_pairs, fixed_paths


class LocalPairMigrationTests(unittest.TestCase):
    def test_nearest_pairs_tie_break_and_midpoint_order(self):
        pool = np.array([[1., 0.], [0., 1.], [2., 0.], [0., 2.]])
        ids = np.array([7, 3, 10, 11])
        pairs, anchors = nearest_pairs(pool, np.zeros(2), np.ones(2), ids, k=3)
        np.testing.assert_array_equal(pairs, [[3, 7], [3, 10], [7, 10]])
        np.testing.assert_allclose(anchors, [[.5, .5], [1., .5], [1.5, 0.]])
        np.testing.assert_array_equal(pool, [[1., 0.], [0., 1.], [2., 0.], [0., 2.]])

    def test_twenty_four_nearest_produce_exact_frozen_count(self):
        pool = np.arange(30 * 2, dtype=float).reshape(30, 2)
        pairs, anchors = nearest_pairs(pool, np.zeros(2), np.ones(2), np.arange(30), k=24)
        self.assertEqual(pairs.shape, (276, 2))
        self.assertEqual(anchors.shape, (276, 2))
        paths, anchor_index, lam = fixed_paths(np.zeros(2), anchors,
                                                [.05, .1, .2, .35, .5, .75, 1.])
        self.assertEqual(paths.shape, (1932, 2))
        np.testing.assert_allclose(lam[:7], [.05, .1, .2, .35, .5, .75, 1.])
        np.testing.assert_array_equal(anchor_index[:7], np.zeros(7, dtype=int))

    def test_paths_exclude_source_identity_and_reject_bad_inputs(self):
        source = np.array([1., 2.])
        anchors = np.array([[3., 4.]])
        paths, ids, lam = fixed_paths(source, anchors, [.5, 1.])
        np.testing.assert_allclose(paths, [[2., 3.], [3., 4.]])
        np.testing.assert_array_equal(source, [1., 2.])
        with self.assertRaises(ValueError):
            fixed_paths(source, anchors, [0., 1.])
        with self.assertRaises(ValueError):
            nearest_pairs(np.array([[1., 2.]]), source, np.ones(2), np.array([1]), k=2)


if __name__ == "__main__":
    unittest.main()
