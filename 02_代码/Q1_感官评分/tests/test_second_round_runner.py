from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from second_round_runner import (  # noqa: E402
    assert_complete_oof,
    load_run_checkpoint,
    outer_indices,
    parse_repeats,
    save_run_checkpoint,
    save_prediction_archive,
)


class SecondRoundRunnerTests(unittest.TestCase):
    def test_e004_config_freezes_three_screening_repeats(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config = yaml.safe_load(
            (
                root
                / "02_代码"
                / "Q1_感官评分"
                / "configs"
                / "E004_ordinal_hgb.yaml"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(config["experiment_id"], "E004")
        self.assertEqual(config["screening_repeats"], [0, 1, 2])
        self.assertEqual(len(config["profiles"]), 3)

    def test_e004_gpu_config_is_separate_from_cpu_experiment(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config = yaml.safe_load(
            (
                root
                / "02_代码"
                / "Q1_感官评分"
                / "configs"
                / "E004_gpu_ordinal_xgb.yaml"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(config["experiment_id"], "E004_GPU")
        self.assertEqual(config["device"], "cuda")
        self.assertEqual(config["screening_repeats"], [0, 1, 2])

    def test_e005_config_freezes_bases_and_weight_grid(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config = yaml.safe_load(
            (
                root
                / "02_代码"
                / "Q1_感官评分"
                / "configs"
                / "E005_continuous_ensemble.yaml"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            config["base_models"],
            ["extra_trees", "hgb", "ridge_quantile", "pls2_quantile"],
        )
        self.assertEqual(config["weight_step"], 0.1)
        self.assertEqual(config["screening_repeats"], [0, 1, 2])

    def test_outer_indices_are_disjoint_and_complete(self) -> None:
        splits = pd.DataFrame(
            {"repeat": [0, 0], "fold": [0, 0], "row_position": [1, 3]}
        )

        train, validation = outer_indices(splits, repeat=0, fold=0, n_samples=5)

        self.assertEqual(set(train) & set(validation), set())
        self.assertEqual(set(train) | set(validation), set(range(5)))

    def test_complete_oof_rejects_missing_slots(self) -> None:
        prediction = np.full((2, 5, 3), np.nan)

        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            assert_complete_oof(prediction)

    def test_repeat_parser_is_sorted_unique_and_bounded(self) -> None:
        self.assertEqual(
            parse_repeats("2,0,1,2", available=list(range(10))), [0, 1, 2]
        )
        with self.assertRaisesRegex(ValueError, "not available"):
            parse_repeats("10", available=list(range(10)))

    def test_prediction_archive_round_trip(self) -> None:
        prediction = np.arange(24, dtype=float).reshape(2, 4, 3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.npz"
            save_prediction_archive(
                path,
                y_true=np.zeros((4, 3)),
                beer=np.arange(4),
                style=np.array(["A", "A", "B", "B"]),
                target_code=np.array(["x", "y", "z"]),
                repeats=np.array([0, 1]),
                fold_assignment=np.zeros((2, 4), dtype=int),
                predictions={"pred_model": prediction},
            )
            archive = np.load(path, allow_pickle=True)

        np.testing.assert_allclose(archive["pred_model"], prediction)
        np.testing.assert_array_equal(archive["repeat"], [0, 1])

    def test_run_checkpoint_round_trip_preserves_partial_oof(self) -> None:
        prediction = np.full((2, 4, 3), np.nan)
        prediction[0, :2] = 1.25
        assignment = np.full((2, 4), -1, dtype=np.int8)
        assignment[0, :2] = 0
        rows = [{"repeat": 0, "fold": 0, "profile": "compact"}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.npz"
            save_run_checkpoint(
                path,
                run_signature="frozen-signature",
                predictions=prediction,
                fold_assignment=assignment,
                parameter_rows=rows,
            )
            restored = load_run_checkpoint(
                path,
                run_signature="frozen-signature",
                prediction_shape=prediction.shape,
                assignment_shape=assignment.shape,
            )

        np.testing.assert_allclose(
            restored["predictions"], prediction, equal_nan=True
        )
        np.testing.assert_array_equal(restored["fold_assignment"], assignment)
        self.assertEqual(restored["parameter_rows"], rows)

    def test_run_checkpoint_rejects_stale_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.npz"
            save_run_checkpoint(
                path,
                run_signature="old-signature",
                predictions=np.full((1, 2, 1), np.nan),
                fold_assignment=np.full((1, 2), -1, dtype=np.int8),
                parameter_rows=[],
            )
            with self.assertRaisesRegex(RuntimeError, "signature"):
                load_run_checkpoint(
                    path,
                    run_signature="new-signature",
                    prediction_shape=(1, 2, 1),
                    assignment_shape=(1, 2),
                )


if __name__ == "__main__":
    unittest.main()
