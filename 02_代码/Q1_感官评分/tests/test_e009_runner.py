from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from run_e009_nnls_stacking import require_e007_audit, fit_outer_nnls


class E009RunnerTests(unittest.TestCase):
    def test_e007_incomplete_or_missing_prevents_new_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "E007"):
                require_e007_audit(path)
            self.assertEqual(list(path.iterdir()), [])
            (path / "q1_run_manifest_E007.json").write_text(json.dumps({
                "status": "complete", "outer_folds": 10, "repeats": list(range(10))}))
            with self.assertRaises(RuntimeError):
                require_e007_audit(path)

    def test_completed_e007_still_requires_frozen_audit_and_no_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "q1_run_manifest_E007.json").write_text(json.dumps({
                "status": "complete", "outer_folds": 30, "repeats": list(range(10))}))
            audit_path = path / "q1_second_round_decision.json"
            audit_path.write_text(json.dumps({"status": "complete", "completed_candidates": ["E006"]}))
            with self.assertRaises(RuntimeError):
                require_e007_audit(path)
            audit_path.write_text(json.dumps({"status": "complete", "completed_candidates": ["E007"],
                                             "selected_experiment": "E007"}))
            with self.assertRaisesRegex(RuntimeError, "promoted"):
                require_e007_audit(path)
            audit_path.write_text(json.dumps({"status": "complete", "completed_candidates": ["E007"],
                                             "selected_experiment": None, "retained_model": "E003"}))
            self.assertEqual(require_e007_audit(path)["retained_model"], "E003")

    def test_outer_fold_uses_fresh_inner_oof_and_no_validation_labels(self):
        rng = np.random.default_rng(90)
        x = rng.normal(size=(30, 4))
        y = np.column_stack([x[:, 0], -x[:, 0] + .2 * x[:, 2]])
        source = {"base_models": ["ridge_quantile"], "ridge_alphas": [1., 10.]}
        result = fit_outer_nnls(
            x, y, x[:4], base_config=source, groups=["same", "same"],
            scopes=["shared", "group", "target"], shrinkage=.5, inner_splits=3, seed=9,
        )
        self.assertEqual(result["inner_prediction_stack"].shape, (30, 2, 1))
        self.assertEqual(result["outer_base_stack"].shape, (4, 2, 1))
        self.assertEqual(set(result["predictions"]), {"shared", "group", "target"})
        self.assertEqual(result["predictions"]["group"].shape, (4, 2))
        self.assertTrue(np.isfinite(result["inner_prediction_stack"]).all())
        self.assertEqual(set(result["inner_fold_assignment"]), {0, 1, 2})

    def test_synthetic_screening_writes_artifacts_and_resumes_without_refitting(self):
        import run_e009_nnls_stacking as runner
        import second_round_runner
        rng = np.random.default_rng(9)
        x = pd.DataFrame(rng.normal(size=(18, 3)), columns=["x", "z", "v"])
        y = pd.DataFrame({"a": x["x"] + rng.normal(size=18) * .1,
                          "b": -x["x"] + rng.normal(size=18) * .1})
        style = pd.Series(["s"] * 18)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attachments = root / "attachments"
            attachments.mkdir()
            pd.DataFrame({"beer": np.arange(18)}).to_csv(attachments / "1.csv", index=False)
            y.to_csv(attachments / "4.csv", index=False)
            registration = yaml.safe_load(runner.CONFIG_PATH.read_text())
            registration["base_models"] = ["ridge_quantile"]
            config_path = root / "E009.yaml"
            config_path.write_text(yaml.safe_dump(registration))
            (root / registration["source_config"]).write_text(yaml.safe_dump({
                "experiment_id": "E005", "base_models": ["ridge_quantile"], "ridge_alphas": [1.]}))
            pd.DataFrame([{"repeat": r, "fold": i % 3, "row_position": i}
                          for r in range(3) for i in range(18)]).to_csv(root / "q1_cv_splits.csv", index=False)
            pd.DataFrame({"target_code": ["a", "b"], "target_group": ["g", "g"]}).to_csv(
                root / "q1_target_dictionary.csv", index=False)
            (root / "q1_run_manifest_E007.json").write_text(json.dumps({
                "status": "complete", "outer_folds": 30, "repeats": list(range(10))}))
            (root / "q1_second_round_decision.json").write_text(json.dumps({
                "status": "complete", "completed_candidates": ["E007"],
                "selected_experiment": None, "retained_model": "E003"}))
            diagnostic_dir = root / "diagnostics_nnls"
            diagnostic_dir.mkdir()
            (diagnostic_dir / "q1_complementarity_manifest.json").write_text(json.dumps({
                "diagnostic_only": True, "n_repeats": 10}))
            pd.DataFrame([{"evaluation": f"repeat_{r}", "r2": 0., "mae": 1.,
                           "spearman": 0., "qwk_proxy": 0., "acc_pm1_proxy": .4}
                          for r in range(3)]).to_csv(root / "q1_extended_metrics_E003.csv", index=False)
            with patch.object(runner, "RESULT_DIR", root), \
                 patch.object(second_round_runner, "RESULT_DIR", root), \
                 patch.object(runner, "ATTACHMENTS", attachments), \
                 patch.object(runner, "CONFIG_PATH", config_path), \
                 patch.object(runner, "load_q1", return_value=(x, y, style)):
                result = runner.run("screening", prior_results=root)
                self.assertEqual(result["outer_folds"], 9)
                self.assertEqual(result["primary_scope"], "group")
                self.assertTrue((root / "q1_screening_decision_E009.json").exists())
                with patch.object(runner, "_fit_base", side_effect=AssertionError("must resume")):
                    runner.run("screening", prior_results=root)
                with np.load(root / "q1_oof_predictions_E009_screening.npz") as archive:
                    self.assertEqual(archive["pred_nnls_group"].shape, (3, 18, 2))


if __name__ == "__main__":
    unittest.main()
