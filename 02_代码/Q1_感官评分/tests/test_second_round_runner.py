from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from second_round_runner import (  # noqa: E402
    assert_complete_oof,
    outer_indices,
    parse_repeats,
    save_prediction_archive,
)


class SecondRoundRunnerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
