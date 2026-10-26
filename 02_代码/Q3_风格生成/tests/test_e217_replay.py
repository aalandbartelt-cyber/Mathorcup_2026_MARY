from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verify_e217_local_pair import reconstruct_locked_spectrum


class E217ReplayTests(unittest.TestCase):
    def test_rebuilds_midpoint_path_and_known_direct_bound(self):
        source = np.array([1., 2., 3.])
        a = np.array([3., 4., 5.])
        b = np.array([5., 6., 7.])
        candidate = reconstruct_locked_spectrum(source, a, b, .5,
            [0], [1.], [True], [10.], ["raw"])
        np.testing.assert_allclose(candidate, [1., 3.5, 4.5])
        np.testing.assert_array_equal(source, [1., 2., 3.])
        with self.assertRaises(ValueError):
            reconstruct_locked_spectrum(source, a, b, 0., [0], [1.], [True], [10.], ["raw"])


if __name__ == "__main__":
    unittest.main()
