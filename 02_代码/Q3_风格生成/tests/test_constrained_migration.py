from pathlib import Path
import sys
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from constrained_migration import (
    concentrations_from_stored,
    empirical_quantile_anchor,
    pareto_mask,
    project_direct_constraints,
)


class ConstrainedMigrationTests(unittest.TestCase):
    def test_storage_scale_conversion_separates_log10_and_raw(self):
        stored = np.array([1.0, 0.25, -2.0])
        result = concentrations_from_stored(stored, ["log10", "raw", "log10"])
        np.testing.assert_allclose(result, [10.0, 0.25, 0.01])

    def test_projection_never_raises_and_preserves_unknown_bounds(self):
        candidate = np.array([2.0, 0.5, -1.0, 0.02])
        source = np.array([10.0, 0.2, np.nan, 0.01])
        known = np.array([True, True, False, True])
        threshold = np.array([19.0, 0.09, 0.2, np.nan])
        scales = ["log10", "raw", "log10", "raw"]
        c1, limits = project_direct_constraints(candidate, source, known, threshold, scales, "C1")
        np.testing.assert_allclose(c1, [1.0, 0.2, -1.0, 0.01])
        self.assertTrue(np.isnan(limits[2]))
        c2, limits = project_direct_constraints(candidate, source, known, threshold, scales, "C2")
        np.testing.assert_allclose(c2, [np.log10(19.0), 0.2, -1.0, 0.02])
        self.assertTrue(np.isnan(limits[3]))
        c3, _ = project_direct_constraints(candidate, source, known, threshold, scales, "C3")
        np.testing.assert_allclose(c3, [1.0, 0.09, -1.0, 0.02])
        self.assertTrue(np.all(c1 <= candidate + 1e-12))
        self.assertTrue(np.all(c2 <= candidate + 1e-12))
        self.assertTrue(np.all(c3 <= candidate + 1e-12))

    def test_empirical_quantile_anchor_is_target_supported(self):
        source = np.array([1.5, 15.0])
        source_reference = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        target_reference = np.array([[100.0, -3.0], [110.0, -2.0], [120.0, -1.0], [130.0, 0.0]])
        anchor = empirical_quantile_anchor(source, source_reference, target_reference)
        self.assertTrue(np.all(anchor >= target_reference.min(0)))
        self.assertTrue(np.all(anchor <= target_reference.max(0)))
        np.testing.assert_allclose(anchor, [115.0, -1.5])

    def test_pareto_front_keeps_only_nondominated_finite_points(self):
        distance = np.array([3.0, 2.0, 1.0, 2.5, np.nan])
        change = np.array([0.0, 1.0, 3.0, 2.0, 0.1])
        np.testing.assert_array_equal(pareto_mask(distance, change), [True, True, True, False, False])


if __name__ == "__main__":
    unittest.main()
