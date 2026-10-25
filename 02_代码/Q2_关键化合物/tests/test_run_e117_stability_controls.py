from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
runner = importlib.import_module("run_e117_stability_controls")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class E117RunnerTests(unittest.TestCase):
    def _ready_fixture(self, base: Path) -> tuple[Path, Path]:
        root = base / "root"
        result = root / "03_结果" / "Q2_关键化合物"
        result.mkdir(parents=True)
        pd.DataFrame(
            {
                "feature_code": [f"x{i}" for i in range(218)],
                "eligible_strict": [True] * 218,
            }
        ).to_csv(result / "q2_candidate_dictionary.csv", index=False)
        write_json(
            result / "q2_run_manifest_E113.json",
            {"status": "complete", "n_outer_folds": 30, "n_features": 218},
        )
        outputs = []
        output_hashes = {}
        for name in ("modules.csv", "fields.csv", "sufficiency.csv"):
            path = result / name
            path.write_text("ok\n", encoding="utf-8")
            outputs.append(str(path))
            output_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(
            result / "q2_run_manifest_E115_final_nnls.json",
            {
                "status": "complete",
                "repeat_ids": list(range(10)),
                "n_outer_folds": 30,
                "n_features_strict": 218,
                "n_targets": 50,
                "q1_retrained": False,
                "nnls_refit": False,
                "hyperparameter_search": False,
                "outputs": outputs,
                "output_sha256": output_hashes,
            },
        )
        q1 = root / "03_结果" / "Q1_感官评分"
        oof = q1 / "publication_final" / "final_model_oof.npz"
        oof.parent.mkdir(parents=True)
        oof.write_bytes(b"frozen-oof")
        write_json(
            q1 / "final_model_registry.json",
            {
                "final_model": "感官组收缩NNLS",
                "oof_path": "03_结果/Q1_感官评分/publication_final/final_model_oof.npz",
                "oof_sha256": hashlib.sha256(oof.read_bytes()).hexdigest(),
            },
        )
        return root, result

    def test_input_audit_accepts_only_complete_formal_e115(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, result = self._ready_fixture(Path(directory))
            audit = runner.audit_inputs(result, root)
            self.assertEqual(audit["status"], "ready")
            self.assertEqual(audit["problems"], [])
            self.assertTrue(audit["e115_output_hashes_match"])

            manifest = result / "q2_run_manifest_E115_final_nnls.json"
            payload = json.loads(manifest.read_text())
            payload["status"] = "pilot_complete"
            write_json(manifest, payload)
            audit = runner.audit_inputs(result, root)
            self.assertEqual(audit["status"], "blocked")
            self.assertTrue(any("pilot/incomplete" in value for value in audit["problems"]))

    def test_input_audit_rejects_tampered_e115_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, result = self._ready_fixture(Path(directory))
            (result / "fields.csv").write_text("tampered\n", encoding="utf-8")
            audit = runner.audit_inputs(result, root)
            self.assertEqual(audit["status"], "blocked")
            self.assertFalse(audit["e115_output_hashes_match"])
            self.assertTrue(any("hash mismatch" in value for value in audit["problems"]))

    def test_input_audit_can_read_frozen_q1_from_separate_canonical_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, result = self._ready_fixture(Path(directory))
            worktree_root = root / ".worktrees" / "q2"
            audit = runner.audit_inputs(result, worktree_root, q1_root=root)
            self.assertEqual(audit["status"], "ready")
            self.assertEqual(audit["q1_registry_root"], str(root))

    def test_nnls_alignment_aggregates_all_fold_conditional_importance(self) -> None:
        conditional = pd.DataFrame(
            {
                "repeat": [0, 1, 0, 1, 0, 1, 0, 1],
                "fold": [0, 0, 0, 0, 0, 0, 0, 0],
                "target_code": ["t1", "t1", "t1", "t1", "t2", "t2", "t2", "t2"],
                "feature_code": ["a", "a", "b", "b", "a", "a", "b", "b"],
                "median_delta_mae": [0.4, 0.2, 0.1, 0.0, -0.1, 0.1, 0.3, 0.5],
            }
        )
        cpss = pd.DataFrame(
            {
                "target_code": ["t1", "t1", "t2", "t2"],
                "feature_code": ["a", "b", "a", "b"],
                "selection_probability": [0.8, 0.2, 0.3, 0.7],
                "selection_count": [8, 2, 3, 7],
                "selection_trials": [10, 10, 10, 10],
            }
        )
        aligned = runner.align_cpss_with_final_nnls(cpss, conditional)
        t1 = aligned.loc[aligned["target_code"].eq("t1")].set_index("feature_code")
        self.assertEqual(int(t1.loc["a", "nnls_predictive_rank"]), 1)
        self.assertAlmostEqual(float(t1.loc["a", "nnls_median_delta_mae"]), 0.3)
        self.assertEqual(int(t1.loc["b", "nnls_predictive_rank"]), 2)

    def test_reviewed_config_still_fails_closed_on_missing_inputs(self) -> None:
        config = runner.load_config()
        self.assertTrue(config["formal_run"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "input gate failed"):
                runner.run_e117(Path(directory), runner.CONFIG_PATH, Path(directory))

    def test_style_adjusted_selector_returns_topk_for_each_target(self) -> None:
        rng = np.random.default_rng(4)
        styles = np.repeat(["a", "b"], 20)
        x = rng.normal(size=(40, 6))
        y = np.column_stack([2 * x[:, 1] + rng.normal(scale=0.05, size=40), -x[:, 4]])
        selected, score = runner.select_style_adjusted_topk(x, y, styles, top_k=2)
        self.assertEqual(score.shape, (6, 2))
        self.assertEqual([len(value) for value in selected], [2, 2])
        self.assertIn(1, selected[0])
        self.assertIn(4, selected[1])

    def test_within_style_permutation_never_crosses_style_values(self) -> None:
        styles = np.asarray(["a", "a", "b", "b"], dtype=object)
        values = np.asarray([[1], [2], [100], [200]])
        changed = runner._permute_within_style(values, styles, np.random.default_rng(8))
        self.assertEqual(set(changed[:2, 0]), {1, 2})
        self.assertEqual(set(changed[2:, 0]), {100, 200})


if __name__ == "__main__":
    unittest.main()
