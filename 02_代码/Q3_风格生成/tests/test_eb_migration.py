from pathlib import Path
import sys
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from eb_migration import EBStyleGeometry, mahalanobis_upper_projection, migration_path


class EBMigrationTests(unittest.TestCase):
    def test_projection_moves_unconstrained_coordinate_by_covariance(self):
        anchor = np.array([2.0, 0.0])
        covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
        projected, dual = mahalanobis_upper_projection(anchor, covariance, [0], [1.0])
        np.testing.assert_allclose(projected, [1.0, -0.5], atol=1e-10)
        np.testing.assert_allclose(dual, [1.0], atol=1e-10)

    def test_projection_handles_two_active_bounds_and_feasible_identity(self):
        covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
        projected, dual = mahalanobis_upper_projection(np.array([2.0, 2.0]), covariance, [0, 1], [1.0, 1.0])
        np.testing.assert_allclose(projected, [1.0, 1.0], atol=1e-10)
        self.assertTrue((dual >= 0).all())
        same, zero = mahalanobis_upper_projection(np.array([0.0, 0.0]), covariance, [0, 1], [1.0, 1.0])
        np.testing.assert_array_equal(same, [0.0, 0.0])
        np.testing.assert_array_equal(zero, [0.0, 0.0])

    def test_path_keeps_original_source_and_strictly_positive_candidates(self):
        source = np.array([2.0, 4.0])
        endpoint = np.array([4.0, 8.0])
        values = migration_path(source, endpoint, [0.05, 0.5, 1.0], include_identity=True)
        np.testing.assert_array_equal(values[0], source)
        np.testing.assert_allclose(values[1], [2.1, 4.2])
        np.testing.assert_array_equal(values[-1], endpoint)
        with self.assertRaises(ValueError):
            migration_path(source, endpoint, [0.0], include_identity=True)

    def test_style_geometry_has_psd_covariance_and_uses_fixed_shrinkage(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(32, 6))
        styles = np.array(["Pils/Lager"] * 8 + ["Tripel"] * 8 + ["Kriek"] * 8 + ["Lambic"] * 8)
        x[styles == "Tripel", 0] += 2.0
        model = EBStyleGeometry().fit(x, styles, mean_kappa=15.0, covariance_kappa=30.0)
        anchor = model.anchor(x[0], "Pils/Lager", "Tripel")
        self.assertEqual(anchor.shape, (6,))
        self.assertGreater(float(anchor[0]), float(x[0, 0]))
        self.assertGreater(np.linalg.eigvalsh(model.covariance("Tripel")).min(), 0.0)
        self.assertLess(np.linalg.norm(anchor - x[0]), 2.0)


if __name__ == "__main__":
    unittest.main()
