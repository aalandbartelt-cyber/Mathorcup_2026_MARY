from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from grouped_multitask import (
    fit_grouped_multitask, evaluate_e116_gate, aggregate_grouped_rankings,
)
from run_e116_grouped_multitask import (
    fit_outer_fold, load_fold_checkpoint, save_fold_checkpoint,
)


class GroupedMultitaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(116)
        cls.x = rng.normal(size=(48, 5))
        cls.x[0, 1] = np.nan
        cls.y = np.column_stack([
            2 * cls.x[:, 0] + rng.normal(scale=.05, size=48),
            -cls.x[:, 0] + rng.normal(scale=.05, size=48),
            cls.x[:, 3] + rng.normal(scale=.05, size=48),
        ])
        cls.kwargs = dict(
            target_codes=["a", "b", "c"],
            groups=[{"name": "pair", "targets": ["a", "b"]}],
            alpha_grid=[.01, .1], l1_ratio_grid=[.5], inner_splits=3,
            random_state=116,
        )

    def test_grouped_and_independent_targets_preserve_output_order(self):
        fitted = fit_grouped_multitask(self.x, self.y, **self.kwargs)
        pred = fitted.predict(self.x)
        self.assertEqual(pred.shape, self.y.shape)
        self.assertEqual(fitted.coefficients.shape, (3, 5))
        self.assertEqual([r["model"] for r in fitted.selection_rows],
                         ["MultiTaskElasticNet", "ElasticNet"])
        self.assertLess(np.mean(np.abs(pred - self.y)), .3)
        self.assertGreater(fitted.coefficients[0, 0], 0)
        self.assertLess(fitted.coefficients[1, 0], 0)

    def test_prediction_cannot_refit_preprocessing_or_change_coefficients(self):
        fitted = fit_grouped_multitask(self.x, self.y, **self.kwargs)
        before = fitted.coefficients.copy()
        medians = fitted.models[0].preprocessor.named_steps["imputer"].statistics_.copy()
        normal = fitted.predict(self.x[:2])
        fitted.predict(np.full((2, 5), 100000.))
        np.testing.assert_array_equal(fitted.coefficients, before)
        np.testing.assert_array_equal(fitted.predict(self.x[:2]), normal)
        np.testing.assert_array_equal(
            fitted.models[0].preprocessor.named_steps["imputer"].statistics_, medians)
        np.testing.assert_allclose(medians, np.nanmedian(self.x, axis=0))

    def test_inner_preprocessing_is_fitted_to_each_training_subset(self):
        fitted = fit_grouped_multitask(self.x, self.y, **self.kwargs)
        for row in fitted.selection_rows:
            self.assertEqual(len(row["candidate_scores"]), 2)
            for candidate in row["candidate_scores"]:
                self.assertEqual(candidate["inner_train_sizes"], [32, 32, 32])
                self.assertEqual(candidate["inner_validation_sizes"], [16, 16, 16])

    def test_inner_imputer_statistics_match_actual_training_rows(self):
        import grouped_multitask
        original = grouped_multitask._fit_group
        observed = []

        def observe(x, y, **kwargs):
            fitted = original(x, y, **kwargs)
            observed.append((len(x), fitted.preprocessor.named_steps["imputer"].statistics_))
            np.testing.assert_allclose(observed[-1][1], np.nanmedian(x, axis=0))
            return fitted

        with patch.object(grouped_multitask, "_fit_group", side_effect=observe):
            fit_grouped_multitask(self.x, self.y, **self.kwargs)
        self.assertTrue(any(size == 32 for size, _ in observed))
        self.assertTrue(any(not np.allclose(median, np.nanmedian(self.x, axis=0))
                            for size, median in observed if size == 32))

    def test_target_rescaling_does_not_change_hyperparameter_selection(self):
        fitted = fit_grouped_multitask(self.x, self.y, **self.kwargs)
        shifted = self.y * [5., 2., 3.] + [4., -3., 9.]
        other = fit_grouped_multitask(self.x, shifted, **self.kwargs)
        np.testing.assert_allclose(
            other.predict(self.x), fitted.predict(self.x) * [5., 2., 3.] + [4., -3., 9.],
            atol=1e-9,
        )

    def test_rejects_overlapping_or_unknown_groups(self):
        for groups in [
            [{"name": "x", "targets": ["a", "b"]},
             {"name": "y", "targets": ["b", "c"]}],
            [{"name": "x", "targets": ["a", "unknown"]}],
        ]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                fit_grouped_multitask(self.x, self.y, **{**self.kwargs, "groups": groups})

    def test_gate_waits_for_stability_and_anchor_evidence(self):
        decision = evaluate_e116_gate(
            baseline_r2=.24, candidate_r2=.24, baseline_mae=.34, candidate_mae=.34,
            baseline_tail_r2=.01, candidate_tail_r2=.02,
            baseline_stability=None, candidate_stability=None,
            baseline_anchors=None, candidate_anchors=None,
        )
        self.assertFalse(decision["promoted"])
        self.assertEqual(decision["status"], "awaiting_stability_and_anchor_audit")

    def test_zero_coefficients_do_not_get_top3_selection_frequency(self):
        rows = pd.DataFrame({
            "repeat": [0, 0, 1, 1], "fold": [0, 0, 0, 0],
            "target_code": ["t"] * 4, "feature_code": ["zero", "signal"] * 2,
            "effect": [0., .4, 0., .3], "abs_effect": [0., .4, 0., .3],
            "fold_rank": [2, 1, 2, 1],
        })
        result = aggregate_grouped_rankings(rows).set_index("feature_code")
        self.assertEqual(result.loc["zero", "top3_frequency"], 0.)
        self.assertEqual(result.loc["zero", "nonzero_frequency"], 0.)
        self.assertEqual(result.loc["signal", "top3_frequency"], 1.)

    def test_gate_requires_every_registered_condition(self):
        good = dict(
            baseline_r2=.24, candidate_r2=.239, baseline_mae=.34, candidate_mae=.341,
            baseline_tail_r2=.01, candidate_tail_r2=.02,
            baseline_stability=.4, candidate_stability=.5,
            baseline_anchors={"a", "b", "c", "d"},
            candidate_anchors={"a", "b", "c", "d"},
        )
        self.assertTrue(evaluate_e116_gate(**good)["promoted"])
        for changed in [{"candidate_r2": .23}, {"candidate_mae": .35},
                        {"candidate_tail_r2": .01}, {"candidate_stability": .4},
                        {"candidate_anchors": {"a", "b", "c"}}]:
            with self.subTest(changed=changed):
                self.assertFalse(evaluate_e116_gate(**{**good, **changed})["promoted"])

    def test_outer_fold_returns_rankings_without_validation_targets(self):
        config = {key: self.kwargs[key] for key in
                  ["alpha_grid", "l1_ratio_grid", "inner_splits", "random_state"]}
        result = fit_outer_fold(
            self.x, self.y, self.x[:4], feature_codes=list("vwxyz"),
            target_codes=self.kwargs["target_codes"], groups=self.kwargs["groups"],
            config=config,
        )
        self.assertEqual(result["prediction"].shape, (4, 3))
        self.assertEqual(len(result["rankings"]), 15)
        self.assertEqual(result["rankings"].groupby("target_code").size().tolist(), [5, 5, 5])

    def test_checkpoint_refuses_signature_or_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold.npz"
            save_fold_checkpoint(path, signature="expected",
                                 prediction=np.ones((4, 3)), coefficients=np.ones((3, 5)),
                                 selection_rows=[{"group": "example"}])
            data = load_fold_checkpoint(path, signature="expected",
                                        prediction_shape=(4, 3), coefficient_shape=(3, 5))
            np.testing.assert_array_equal(data["prediction"], np.ones((4, 3)))
            with self.assertRaises(RuntimeError):
                load_fold_checkpoint(path, signature="wrong",
                                     prediction_shape=(4, 3), coefficient_shape=(3, 5))
            with self.assertRaises(RuntimeError):
                load_fold_checkpoint(path, signature="expected",
                                     prediction_shape=(3, 3), coefficient_shape=(3, 5))

    def test_runner_checks_prior_experiments_before_creating_outputs(self):
        import run_e116_grouped_multitask as runner
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(runner, "RESULT_DIR", output):
                with self.assertRaisesRegex(RuntimeError, "prior completed E113"):
                    runner.run_e116()
            self.assertEqual(list(output.iterdir()), [])

    def test_checkpoint_rejects_nonfinite_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold.npz"
            save_fold_checkpoint(path, signature="expected", prediction=np.full((4, 3), np.nan),
                                 coefficients=np.ones((3, 5)), selection_rows=[])
            with self.assertRaises(RuntimeError):
                load_fold_checkpoint(path, signature="expected",
                                     prediction_shape=(4, 3), coefficient_shape=(3, 5))


if __name__ == "__main__":
    unittest.main()
