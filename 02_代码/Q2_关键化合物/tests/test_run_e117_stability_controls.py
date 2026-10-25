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
        for name in ("modules.csv", "fields.csv", "sufficiency.csv"):
            path = result / name
            path.write_text("ok\n", encoding="utf-8")
            outputs.append(str(path))
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

            manifest = result / "q2_run_manifest_E115_final_nnls.json"
            payload = json.loads(manifest.read_text())
            payload["status"] = "pilot_complete"
            write_json(manifest, payload)
            audit = runner.audit_inputs(result, root)
            self.assertEqual(audit["status"], "blocked")
            self.assertTrue(any("pilot/incomplete" in value for value in audit["problems"]))

    def test_committed_config_is_fail_closed(self) -> None:
        config = runner.load_config()
        self.assertFalse(config["formal_run"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "formal E117 is disabled"):
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
