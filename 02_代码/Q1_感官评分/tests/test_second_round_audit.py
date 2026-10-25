from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from audit_q1_second_round import (  # noqa: E402
    bottom_quartile_target_summary,
    compare_full_frames,
    compare_screening_frames,
    paired_beer_bootstrap,
    promotion_decision,
    screening_decision,
    select_final_candidate,
    style_mae_table,
    target_delta_table,
    worst_style_summary,
)

import pandas as pd
import audit_q1_second_round as audit_module


class SecondRoundAuditTests(unittest.TestCase):
    def test_target_delta_table_aligns_targets_and_metric_directions(self) -> None:
        baseline = pd.DataFrame(
            {
                "target_code": ["a", "b", "a", "b"],
                "r2": [0.1, 0.2, 0.1, 0.2],
                "mae": [0.4, 0.3, 0.4, 0.3],
                "spearman": [0.2, 0.3, 0.2, 0.3],
                "qwk_proxy": [0.2, 0.3, 0.2, 0.3],
                "acc_pm1_proxy": [0.5, 0.6, 0.5, 0.6],
            }
        )
        candidate = baseline.copy()
        candidate["r2"] += 0.1
        candidate["mae"] -= 0.1

        result = target_delta_table(baseline, candidate)

        self.assertEqual(result["target_code"].tolist(), ["a", "b"])
        self.assertTrue(np.allclose(result["delta_r2"], 0.1))
        self.assertTrue(np.allclose(result["delta_mae"], -0.1))

    def test_e007_screening_audit_writes_standard_outputs(self) -> None:
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
                    "pred_shrunk_gate:repeat_0",
                    "pred_shrunk_gate:repeat_1",
                    "pred_shrunk_gate:repeat_2",
                ],
                "r2": [0.21, 0.22, 0.23],
                "mae": [0.34, 0.34, 0.34],
                "spearman": [0.41, 0.41, 0.41],
                "qwk_proxy": [0.40, 0.40, 0.40],
                "acc_pm1_proxy": [0.55, 0.55, 0.55],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            baseline.to_csv(result_dir / "q1_extended_metrics_E003.csv", index=False)
            candidate.to_csv(
                result_dir / "q1_extended_metrics_E007_screening.csv", index=False
            )
            with patch.object(audit_module, "RESULT_DIR", result_dir):
                summary, payload = audit_module.audit_e007_screening()

            self.assertTrue(bool(summary.loc[0, "promote_to_full"]))
            self.assertTrue(payload["decisions"]["E007"]["promote_to_full"])
            self.assertTrue((result_dir / "q1_screening_comparison_E007.csv").exists())
            self.assertTrue((result_dir / "q1_screening_by_repeat_E007.csv").exists())
            self.assertTrue((result_dir / "q1_screening_decision_E007.json").exists())

    def test_e007_full_audit_writes_standard_outputs(self) -> None:
        repeats = list(range(10))
        baseline = pd.DataFrame(
            {
                "evaluation": [f"repeat_{value}" for value in repeats],
                "r2": [0.23] * 10,
                "mae": [0.343] * 10,
                "spearman": [0.42] * 10,
                "qwk_proxy": [0.40] * 10,
                "acc_pm1_proxy": [0.56] * 10,
            }
        )
        candidate = pd.DataFrame(
            {
                "evaluation": [
                    f"pred_shrunk_gate:repeat_{value}" for value in repeats
                ],
                "r2": [0.24] * 10,
                "mae": [0.34] * 10,
                "spearman": [0.43] * 10,
                "qwk_proxy": [0.41] * 10,
                "acc_pm1_proxy": [0.58] * 10,
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            baseline.to_csv(result_dir / "q1_extended_metrics_E003.csv", index=False)
            candidate.to_csv(result_dir / "q1_extended_metrics_E007.csv", index=False)
            with patch.object(audit_module, "RESULT_DIR", result_dir):
                summary, payload = audit_module.audit_e007_full()

            self.assertTrue(bool(summary.loc[0, "promoted"]))
            self.assertTrue(payload["decision"]["promoted"])
            self.assertTrue((result_dir / "q1_full_comparison_E007.csv").exists())
            self.assertTrue((result_dir / "q1_full_by_repeat_E007.csv").exists())
            self.assertTrue((result_dir / "q1_full_decision_E007.json").exists())

    def test_final_audit_writes_comparison_bootstrap_tail_and_style_outputs(self) -> None:
        repeats = list(range(10))
        metrics_baseline = pd.DataFrame(
            {
                "evaluation": [f"repeat_{value}" for value in repeats],
                "r2": [0.23] * 10,
                "mae": [0.343] * 10,
                "spearman": [0.42] * 10,
                "qwk_proxy": [0.40] * 10,
                "acc_pm1_proxy": [0.56] * 10,
            }
        )
        metrics_candidate = pd.DataFrame(
            {
                "evaluation": [
                    f"pred_shrunk_gate:repeat_{value}" for value in repeats
                ],
                "r2": [0.24] * 10,
                "mae": [0.34] * 10,
                "spearman": [0.43] * 10,
                "qwk_proxy": [0.41] * 10,
                "acc_pm1_proxy": [0.58] * 10,
            }
        )
        target_rows_baseline = []
        target_rows_candidate = []
        for repeat in repeats:
            for index, target in enumerate(["a", "b", "c", "d"]):
                target_rows_baseline.append(
                    {
                        "evaluation": f"repeat_{repeat}",
                        "target_code": target,
                        "r2": 0.1 + 0.1 * index,
                        "mae": 0.4 - 0.05 * index,
                        "spearman": 0.2 + 0.1 * index,
                        "qwk_proxy": 0.2 + 0.1 * index,
                        "acc_pm1_proxy": 0.5 + 0.05 * index,
                    }
                )
                target_rows_candidate.append(
                    {
                        "evaluation": f"pred_shrunk_gate:repeat_{repeat}",
                        "target_code": target,
                        "r2": 0.2 + 0.1 * index,
                        "mae": 0.3 - 0.04 * index,
                        "spearman": 0.3 + 0.1 * index,
                        "qwk_proxy": 0.3 + 0.1 * index,
                        "acc_pm1_proxy": 0.6 + 0.05 * index,
                    }
                )
        y_true = np.arange(16, dtype=float).reshape(4, 4)
        baseline_prediction = np.stack([y_true + 1.0] * 10)
        candidate_prediction = np.stack([y_true + 0.5] * 10)
        common_npz = {
            "y_true": y_true,
            "beer": np.arange(4),
            "style": np.array(["A", "A", "B", "B"]),
            "target_code": np.array(["a", "b", "c", "d"]),
            "repeat": np.arange(10),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            metrics_baseline.to_csv(
                result_dir / "q1_extended_metrics_E003.csv", index=False
            )
            metrics_candidate.to_csv(
                result_dir / "q1_extended_metrics_E007.csv", index=False
            )
            pd.DataFrame(target_rows_baseline).to_csv(
                result_dir / "q1_extended_metrics_by_target_E003.csv", index=False
            )
            pd.DataFrame(target_rows_candidate).to_csv(
                result_dir / "q1_extended_metrics_by_target_E007.csv", index=False
            )
            np.savez_compressed(
                result_dir / "q1_oof_predictions_E003.npz",
                **common_npz,
                pred_extra_trees=baseline_prediction,
            )
            np.savez_compressed(
                result_dir / "q1_oof_predictions_E007.npz",
                **common_npz,
                pred_shrunk_gate=candidate_prediction,
            )
            with patch.object(audit_module, "RESULT_DIR", result_dir):
                payload = audit_module.audit_second_round_final(
                    n_bootstraps=100, random_state=17
                )

            self.assertEqual(payload["selected_experiment"], "E007")
            self.assertEqual(payload["retained_model"], "E007")
            for filename in [
                "q1_second_round_comparison.csv",
                "q1_second_round_by_repeat.csv",
                "q1_second_round_bootstrap.csv",
                "q1_second_round_tail_targets.csv",
                "q1_second_round_target_deltas.csv",
                "q1_second_round_style_mae.csv",
                "q1_second_round_worst_style.csv",
                "q1_second_round_decision.json",
            ]:
                self.assertTrue((result_dir / filename).exists(), filename)

    def test_paired_beer_bootstrap_detects_uniform_improvement(self) -> None:
        y_true = np.arange(48, dtype=float).reshape(12, 4)
        baseline = y_true[::-1].copy()
        candidate = y_true.copy()

        result = paired_beer_bootstrap(
            y_true,
            baseline,
            candidate,
            n_bootstraps=500,
            random_state=17,
        )

        self.assertEqual(set(result["metric"]), {"mae", "absolute_rank_error"})
        self.assertTrue((result["delta_candidate_minus_baseline"] < 0).all())
        self.assertTrue((result["ci_high"] <= 0).all())
        self.assertTrue((result["improvement_probability"] >= 0.99).all())

    def test_bottom_quartile_uses_baseline_target_r2_only(self) -> None:
        baseline = pd.DataFrame(
            {
                "target_code": ["a", "b", "c", "d"] * 2,
                "repeat": [0] * 4 + [1] * 4,
                "r2": [0.1, 0.2, 0.3, 0.4] * 2,
                "mae": [0.4, 0.3, 0.2, 0.1] * 2,
            }
        )
        candidate = baseline.copy()
        candidate.loc[candidate["target_code"].eq("a"), "r2"] = 0.25
        candidate.loc[candidate["target_code"].eq("a"), "mae"] = 0.25

        result = bottom_quartile_target_summary(baseline, candidate)

        self.assertEqual(result["bottom_targets"], ["a"])
        self.assertAlmostEqual(result["baseline_r2"], 0.1)
        self.assertAlmostEqual(result["candidate_r2"], 0.25)
        self.assertAlmostEqual(result["candidate_mae"], 0.25)

    def test_style_mae_table_preserves_repeat_and_style_pairing(self) -> None:
        y_true = np.zeros((4, 2), dtype=float)
        styles = np.array(["A", "A", "B", "B"])
        baseline = np.stack(
            [np.ones((4, 2)), np.full((4, 2), 2.0)], axis=0
        )
        candidate = baseline * 0.5

        result = style_mae_table(y_true, styles, baseline, candidate)

        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["style"]), {"A", "B"})
        self.assertTrue((result["candidate_mae"] < result["baseline_mae"]).all())
        self.assertTrue((result["delta_mae"] < 0).all())

    def test_worst_style_is_selected_from_candidate_mean_mae(self) -> None:
        table = pd.DataFrame(
            {
                "repeat": [0, 0, 1, 1],
                "style": ["A", "B", "A", "B"],
                "n_beers": [2, 2, 2, 2],
                "baseline_mae": [0.3, 0.4, 0.3, 0.4],
                "candidate_mae": [0.2, 0.5, 0.2, 0.6],
                "delta_mae": [-0.1, 0.1, -0.1, 0.2],
            }
        )

        result = worst_style_summary(table)

        self.assertEqual(result["worst_style"], "B")
        self.assertAlmostEqual(result["candidate_mae"], 0.55)
        self.assertAlmostEqual(result["baseline_mae"], 0.4)

    def test_final_selection_excludes_candidates_that_miss_a_gate(self) -> None:
        decisions = {
            "E006": {
                "promoted": False,
                "candidate_means": {
                    "r2": 0.26, "mae": 0.33, "spearman": 0.44,
                    "qwk_proxy": 0.43, "acc_pm1_proxy": 0.56,
                },
            },
            "E007": {
                "promoted": True,
                "candidate_means": {
                    "r2": 0.25, "mae": 0.335, "spearman": 0.435,
                    "qwk_proxy": 0.42, "acc_pm1_proxy": 0.58,
                },
            },
        }

        self.assertEqual(select_final_candidate(decisions), "E007")
        decisions["E007"]["promoted"] = False
        self.assertIsNone(select_final_candidate(decisions))

    def test_full_frame_comparison_requires_ten_aligned_repeats(self) -> None:
        repeats = list(range(10))
        baseline = pd.DataFrame(
            {
                "evaluation": [f"repeat_{value}" for value in repeats],
                "r2": [0.23] * 10,
                "mae": [0.343] * 10,
                "spearman": [0.42] * 10,
                "qwk_proxy": [0.40] * 10,
                "acc_pm1_proxy": [0.56] * 10,
            }
        )
        candidate = pd.DataFrame(
            {
                "evaluation": [f"pred_blend:repeat_{value}" for value in repeats],
                "r2": [0.24] * 10,
                "mae": [0.34] * 10,
                "spearman": [0.43] * 10,
                "qwk_proxy": [0.41] * 10,
                "acc_pm1_proxy": [0.58] * 10,
            }
        )

        summary, paired, decision = compare_full_frames(
            baseline,
            candidate,
            experiment_id="E005",
            model="continuous_blend",
            candidate_prefix="pred_blend:repeat_",
        )

        self.assertEqual(summary["n_repeats"], 10)
        self.assertTrue(summary["promoted"])
        self.assertEqual(paired["repeat"].tolist(), repeats)
        self.assertEqual(decision["repeat_win_counts"]["r2"], 10)

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
