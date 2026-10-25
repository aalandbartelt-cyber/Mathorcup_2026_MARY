from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from target_partial_effects import (  # noqa: E402
    aggregate_partial_effect_rankings,
    ledoit_wolf_target_partial,
    load_target_codes,
    load_target_groups,
    rank_signed_effects,
)


E116_CONFIG_PATH = SRC.parent / "configs" / "E116_grouped_multitask.yaml"


class TargetPartialEffectTests(unittest.TestCase):
    def test_target_groups_do_not_overlap_and_cover_registered_targets(self) -> None:
        groups = load_target_groups(E116_CONFIG_PATH)
        flattened = [target for group in groups for target in group["targets"]]

        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(set(flattened).issubset(set(load_target_codes())))

    def test_direct_signal_has_largest_partial_effect(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(size=(160, 5))
        y = 2.0 * x[:, 2] + rng.normal(scale=0.2, size=160)

        effect = ledoit_wolf_target_partial(x, y)

        self.assertEqual(int(np.argmax(np.abs(effect))), 2)

    def test_signed_rank_is_deterministic(self) -> None:
        effect = np.array([0.5, -0.5, 0.1])

        first = rank_signed_effects(effect, ["b", "a", "c"])
        second = rank_signed_effects(effect, ["b", "a", "c"])

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["feature_code"].tolist(), ["a", "b", "c"])
        self.assertEqual(first["direction"].tolist(), ["negative", "positive", "positive"])

    def test_aggregate_reports_top_frequencies_and_sign_consistency(self) -> None:
        rows = pd.DataFrame(
            {
                "repeat": [0, 1, 0, 1],
                "fold": [0, 0, 0, 0],
                "target_code": ["t", "t", "t", "t"],
                "feature_code": ["a", "a", "b", "b"],
                "effect": [0.5, 0.4, -0.2, 0.1],
                "abs_effect": [0.5, 0.4, 0.2, 0.1],
                "fold_rank": [1, 2, 4, 8],
            }
        )

        result = aggregate_partial_effect_rankings(rows)

        first = result.loc[result["feature_code"].eq("a")].iloc[0]
        second = result.loc[result["feature_code"].eq("b")].iloc[0]
        self.assertEqual(first["partial_effect_rank"], 1)
        self.assertAlmostEqual(first["top3_frequency"], 1.0)
        self.assertAlmostEqual(first["sign_consistency"], 1.0)
        self.assertAlmostEqual(second["sign_consistency"], 0.5)


if __name__ == "__main__":
    unittest.main()
