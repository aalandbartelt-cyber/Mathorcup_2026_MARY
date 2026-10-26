from pathlib import Path
import sys
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from eb_box_migration import mahalanobis_box_projection, project_stored_box
from eb_migration import EBStyleGeometry


class EBBoxMigrationTests(unittest.TestCase):
    def test_lower_bound_moves_other_field_by_covariance(self):
        anchor = np.array([-1.0, 0.0])
        covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
        result, dual = mahalanobis_box_projection(anchor, covariance, [0], [0.0], [np.inf])
        np.testing.assert_allclose(result, [0.0, 0.5], atol=1e-10)
        self.assertLess(dual[0], 0.0)

    def test_two_sided_projection_and_identity(self):
        covariance = np.eye(2)
        result, dual = mahalanobis_box_projection(np.array([-1.0, 2.0]), covariance, [0, 1], [0.0, 0.0], [1.0, 1.0])
        np.testing.assert_array_equal(result, [0.0, 1.0])
        np.testing.assert_allclose(dual, [-1.0, 1.0])
        same, zero = mahalanobis_box_projection(np.array([0.3, 0.7]), covariance, [0, 1], [0.0, 0.0], [1.0, 1.0])
        np.testing.assert_array_equal(same, [0.3, 0.7])
        np.testing.assert_array_equal(zero, [0.0, 0.0])

    def test_stored_projection_obeys_raw_lower_and_known_upper(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(32, 4))
        styles = np.array(["Pils/Lager"] * 8 + ["Tripel"] * 8 + ["Kriek"] * 8 + ["Lambic"] * 8)
        model = EBStyleGeometry().fit(x, styles)
        anchor = np.array([-2.0, 1.0, 0.0, 0.0])
        candidate, dual = project_stored_box(model, anchor, "Tripel", [0, 1], [0.0, -np.inf], [0.5, 0.5])
        self.assertGreaterEqual(candidate[0], -1e-9)
        self.assertLessEqual(candidate[0], 0.5 + 1e-9)
        self.assertLessEqual(candidate[1], 0.5 + 1e-9)
        self.assertEqual(len(dual), 2)


if __name__ == "__main__":
    unittest.main()
