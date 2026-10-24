from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from audit_q1_second_round import (  # noqa: E402
    compare_screening_frames,
    promotion_decision,
    screening_decision,
)

import pandas as pd


class SecondRoundAuditTests(unittest.TestCase):
    def test_screening_frame_comparison_aligns_repeats(self) -> None:
        baseline = pd.DataFrame(
            {
                "evaluation": ["repeat_0", "repeat_1", "repeat_2"],
                "r2": [0.20, 0.21, 0.22],
                "mae": [0.35, 0.35, 0.35],
                "spearman": [0.40, 0.40, 0.40],
                "qwk_proxy": [0.39, 0.39, 0.39],
                "acc_pm1_proxy": [0.54, 0.54, 0.54],
            }
        )
        candidate = pd.DataFrame(
            {
                "evaluation": [
                    "pred_blend:repeat_0",
                    "pred_blend:repeat_1",
                    "pred_blend:repeat_2",
                ],
                "r2": [0.21, 0.22, 0.23],
                "mae": [0.34, 0.34, 0.34],
                "spearman": [0.41, 0.41, 0.41],
                "qwk_proxy": [0.40, 0.40, 0.40],
                "acc_pm1_proxy": [0.55, 0.55, 0.55],
            }
        )

        summary, paired, decision = compare_screening_frames(
            baseline,
            candidate,
            experiment_id="E005",
            model="continuous_blend",
            candidate_prefix="pred_blend:repeat_",
        )

        self.assertEqual(summary["n_repeats"], 3)
        self.assertTrue(summary["promote_to_full"])
        self.assertEqual(paired["repeat"].tolist(), [0, 1, 2])
        self.assertEqual(decision["repeat_ids"], [0, 1, 2])

    def test_promotion_requires_all_metrics_and_seven_repeats(self) -> None:
        means = {
            "r2": 0.24,
            "mae": 0.34,
            "spearman": 0.43,
            "qwk_proxy": 0.41,
            "acc_pm1_proxy": 0.58,
        }
        counts = {name: 7 for name in means}

        self.assertTrue(promotion_decision(means, counts)["promoted"])
        counts["acc_pm1_proxy"] = 6
        self.assertFalse(promotion_decision(means, counts)["promoted"])

    def test_hard_thresholds_are_strict(self) -> None:
        means = {
            "r2": 0.2359,
            "mae": 0.3422,
            "spearman": 0.4260,
            "qwk_proxy": 0.4026,
            "acc_pm1_proxy": 0.5734,
        }
        counts = {name: 10 for name in means}

        decision = promotion_decision(means, counts)

        self.assertFalse(decision["promoted"])
        self.assertFalse(any(decision["mean_threshold_flags"].values()))

    def test_screening_requires_four_improvements_and_ordered_gain(self) -> None:
        baseline = {
            "r2": 0.23,
            "mae": 0.343,
            "spearman": 0.41,
            "qwk_proxy": 0.40,
            "acc_pm1_proxy": 0.55,
        }
        candidate = {
            "r2": 0.231,
            "mae": 0.342,
            "spearman": 0.42,
            "qwk_proxy": 0.401,
            "acc_pm1_proxy": 0.54,
        }

        decision = screening_decision(candidate, baseline)

        self.assertTrue(decision["promote_to_full"])
        candidate["spearman"] = 0.40
        self.assertFalse(screening_decision(candidate, baseline)["promote_to_full"])


if __name__ == "__main__":
    unittest.main()
