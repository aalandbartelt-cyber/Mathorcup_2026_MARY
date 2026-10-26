from pathlib import Path
import sys
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from floor_aware_copula import FloorAwareStyleCopula


class FloorAwareCopulaTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(44)
        self.x = rng.normal(size=(40, 4))
        self.x[:12, 0] = -2.013157381
        # This repeated minimum is intentionally not registered as a floor.
        self.x[:20, 1] = self.x[:, 1].min()
        self.styles = np.array(["Tripel"] * 20 + ["Lambic"] * 20)
        self.floor = {0: {"value": -2.013157381, "tolerance": 1e-9, "prior_strength": 10.0}}

    def test_registered_floor_is_modelled_and_unregistered_ties_are_not(self):
        model = FloorAwareStyleCopula().fit(self.x, self.styles, floor_specs=self.floor)
        sample = model.generate("Tripel", 1000, seed=7)
        values = sample["values"]
        self.assertTrue(np.isfinite(values).all())
        self.assertIn(0, sample["floor_fraction"])
        self.assertNotIn(1, sample["floor_fraction"])
        self.assertGreater(sample["floor_fraction"][0], 0.0)
        self.assertLess(sample["floor_fraction"][0], 1.0)
        self.assertTrue(np.all(values[:, 0] >= -2.013157381 - 1e-9))
        nonfloor = ~np.isclose(values[:, 0], -2.013157381, atol=1e-9, rtol=0)
        self.assertTrue(np.all(values[nonfloor, 0] > -2.013157381 + 1e-9))

    def test_seed_and_state_restore_are_exact(self):
        model = FloorAwareStyleCopula().fit(self.x, self.styles, floor_specs=self.floor)
        first = model.generate("Lambic", 31, seed=19)
        second = model.generate("Lambic", 31, seed=19)
        np.testing.assert_array_equal(first["values"], second["values"])
        restored = FloorAwareStyleCopula.from_state(model.state_dict())
        third = restored.generate("Lambic", 31, seed=19)
        np.testing.assert_array_equal(first["values"], third["values"])

    def test_constants_and_invalid_floor_specs_are_safe(self):
        x = self.x.copy()
        x[:, 3] = 2.5
        model = FloorAwareStyleCopula().fit(x, self.styles, floor_specs=self.floor)
        sample = model.generate("Tripel", 50, seed=3)
        self.assertTrue(np.all(sample["values"][:, 3] == 2.5))
        with self.assertRaises(ValueError):
            FloorAwareStyleCopula().fit(self.x, self.styles, floor_specs={8: self.floor[0]})
        with self.assertRaises(ValueError):
            FloorAwareStyleCopula().fit(self.x, self.styles, floor_specs={0: {"value": np.nan}})


if __name__ == "__main__":
    unittest.main()
