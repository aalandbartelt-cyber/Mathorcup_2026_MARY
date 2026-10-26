from pathlib import Path
import sys
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from floor_aware_copula import FloorAwareStyleCopula
from compact_e214_states import compact_state


class E214CompactionTests(unittest.TestCase):
    def test_only_registered_styles_survive_without_changing_draws(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(24, 4))
        labels = np.array(["Tripel"] * 6 + ["Kriek"] * 6 + ["Lambic"] * 6 + ["Amber"] * 6)
        x[:4, 0] = -2.013157381
        model = FloorAwareStyleCopula().fit(x, labels, floor_specs={0: {"value": -2.013157381}})
        state = {"model": model.state_dict(), "features": ["X4VG", "a", "b", "c"], "imputer_statistics": x.mean(0)}
        compact = compact_state(state)
        self.assertEqual(set(compact["model"]["models"]), {"Tripel", "Kriek", "Lambic"})
        restored = FloorAwareStyleCopula.from_state(compact["model"])
        for style in ["Tripel", "Kriek", "Lambic"]:
            np.testing.assert_array_equal(model.generate(style, 10, seed=9)["values"], restored.generate(style, 10, seed=9)["values"])
        self.assertEqual(set(state["model"]["models"]), {"Tripel", "Kriek", "Lambic", "Amber"})


if __name__ == "__main__":
    unittest.main()
