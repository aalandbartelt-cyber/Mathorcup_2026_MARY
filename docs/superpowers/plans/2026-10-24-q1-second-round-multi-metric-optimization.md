# Q1 Second-Round Multi-Metric Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and evaluate a leakage-safe continuous–ordinal ensemble that improves Q1 R², MAE, Spearman, proxy QWK, and proxy ACC±1 on the frozen 10×3 outer OOF protocol.

**Architecture:** Add reusable matrix evaluation, cumulative-ordinal HGB, continuous HGB, constrained convex blending, and nested cross-fitting components. Run E004–E006 in a three-repeat screening stage, promote only viable combinations to all ten repeats, and invoke E007 shrinkage gating only if the global blend misses a hard threshold.

**Tech Stack:** Python 3.10, NumPy 2.2, pandas 2.3, SciPy 1.15, scikit-learn 1.7, YAML, unittest, CPU-only execution.

---

## File map

| Path | Responsibility |
|---|---|
| 02_代码/Q1_感官评分/src/q1_metrics.py | Reusable continuous and ordered-proxy metrics |
| 02_代码/Q1_感官评分/src/ordered_models.py | Training-only anchors, cumulative HGB, probability projection and decoding |
| 02_代码/Q1_感官评分/src/ensemble_models.py | Continuous HGB, convex blending and target-weight shrinkage |
| 02_代码/Q1_感官评分/src/second_round_runner.py | Frozen splits, inner OOF, completeness checks and artifacts |
| 02_代码/Q1_感官评分/src/run_e004_ordinal_hgb.py | E004 screening/full runner |
| 02_代码/Q1_感官评分/src/run_e005_continuous_ensemble.py | E005 screening/full runner |
| 02_代码/Q1_感官评分/src/run_e006_dual_route.py | E006 screening/full runner and gate |
| 02_代码/Q1_感官评分/src/run_e007_shrunk_gating.py | Conditional E007 runner |
| 02_代码/Q1_感官评分/src/audit_q1_second_round.py | Paired comparison, tail audit and final decision |
| 02_代码/Q1_感官评分/tests/test_q1_metrics.py | Metric equivalence |
| 02_代码/Q1_感官评分/tests/test_ordered_models.py | Ordinal model invariants |
| 02_代码/Q1_感官评分/tests/test_ensemble_models.py | Blend invariants |
| 02_代码/Q1_感官评分/tests/test_second_round_runner.py | Split and artifact contract |
| docs/analysis/2026初赛/Q1_二轮优化报告.md | Evidence-backed decision report |

## Task 1: Extract experiment-independent metric functions

**Files:**
- Create: 02_代码/Q1_感官评分/src/q1_metrics.py
- Modify: 02_代码/Q1_感官评分/src/compute_q1_extended_metrics.py
- Create: 02_代码/Q1_感官评分/tests/test_q1_metrics.py

- [x] **Step 1: Write the failing metric tests**

~~~python
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from q1_metrics import matrix_metrics, ordered_proxy_metrics

class Q1MetricTests(unittest.TestCase):
    def test_matrix_metrics_macro_average(self):
        y = np.column_stack([np.arange(14), np.arange(14)[::-1]]).astype(float)
        metrics = matrix_metrics(y, y.copy())
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0)
        self.assertAlmostEqual(metrics["qwk_proxy"], 1.0)
        self.assertAlmostEqual(metrics["acc_pm1_proxy"], 1.0)

    def test_proxy_is_positive_affine_invariant(self):
        y = np.linspace(-2.0, 2.0, 70)
        self.assertEqual(
            ordered_proxy_metrics(y, y),
            ordered_proxy_metrics(y, 3.0 * y + 9.0),
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            matrix_metrics(np.zeros((5, 2)), np.zeros((5, 3)))
~~~

- [x] **Step 2: Verify RED**

~~~bash
conda run -n mathorcup python -m unittest \
  '02_代码/Q1_感官评分/tests/test_q1_metrics.py' -v
~~~

Expected: import failure for q1_metrics.

- [x] **Step 3: Implement the public metric API**

Move the current scalar implementations from compute_q1_extended_metrics.py and add the following matrix layer; the moved scalar functions retain their current bodies exactly:

~~~python
METRIC_COLUMNS = [
    "r2", "mae", "rmse", "spearman", "qwk_proxy", "acc_proxy",
    "acc_pm1_proxy", "level_mae_proxy", "level_spearman_proxy",
]

def matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)
    if true_array.shape != pred_array.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if true_array.ndim != 2:
        raise ValueError("prediction matrices must be two-dimensional")
    rows = [target_metrics(true_array[:, j], pred_array[:, j])
            for j in range(true_array.shape[1])]
    return {name: float(np.nanmean([row[name] for row in rows]))
            for name in METRIC_COLUMNS}

def evaluate_prediction_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                               target_codes: np.ndarray, experiment_id: str,
                               evaluation: str) -> pd.DataFrame:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    return pd.DataFrame([
        {"experiment_id": experiment_id, "evaluation": evaluation,
         "target_code": str(code),
         **target_metrics(y_true[:, j], y_pred[:, j])}
        for j, code in enumerate(target_codes)
    ])

def summarise_target_metrics(by_target: pd.DataFrame) -> pd.DataFrame:
    summary = by_target.groupby(
        ["experiment_id", "evaluation"], sort=False
    )[METRIC_COLUMNS].mean().reset_index()
    summary.insert(2, "n_targets", by_target["target_code"].nunique())
    return summary
~~~

Preserve the BZD seven-bin definition exactly and import all public functions from q1_metrics.py in the E003 CLI.

- [x] **Step 4: Verify GREEN and frozen E003 equivalence**

~~~bash
conda run -n mathorcup python -m unittest \
  '02_代码/Q1_感官评分/tests/test_q1_metrics.py' \
  '02_代码/Q1_感官评分/tests/test_ordered_metrics.py' -v
conda run -n mathorcup python \
  '02_代码/Q1_感官评分/src/compute_q1_extended_metrics.py'
git diff --exit-code -- \
  '03_结果/Q1_感官评分/q1_extended_metrics_E003.csv' \
  '03_结果/Q1_感官评分/q1_extended_metrics_by_target_E003.csv'
~~~

Expected: tests pass and E003 CSV files have no diff.

- [x] **Step 5: Commit**

~~~bash
git add '02_代码/Q1_感官评分/src/q1_metrics.py' \
  '02_代码/Q1_感官评分/src/compute_q1_extended_metrics.py' \
  '02_代码/Q1_感官评分/tests/test_q1_metrics.py'
git commit -m 'refactor(q1): extract reusable multi-metric evaluation'
~~~

## Task 2: Implement cumulative-ordinal HGB primitives

**Files:**
- Create: 02_代码/Q1_感官评分/src/ordered_models.py
- Create: 02_代码/Q1_感官评分/tests/test_ordered_models.py

- [x] **Step 1: Write failing tests**

~~~python
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from ordered_models import (
    decode_expected_level,
    expected_level_to_continuous,
    fit_predict_cumulative_hgb,
    project_survival_probabilities,
    training_quantile_anchors,
)

class OrderedModelTests(unittest.TestCase):
    def test_projection_is_nonincreasing(self):
        raw = np.array([[[0.2, 0.9, 0.4, 1.2, -0.1, 0.3]]])
        projected = project_survival_probabilities(raw)
        self.assertTrue(((projected >= 0) & (projected <= 1)).all())
        self.assertTrue((np.diff(projected, axis=-1) <= 1e-12).all())

    def test_decode_is_monotone(self):
        survival = np.array([
            [[1, 1, 1, 0, 0, 0]],
            [[1, 1, 1, 1, 1, 0]],
        ], dtype=float)
        levels = decode_expected_level(survival)
        anchors = np.arange(7, dtype=float).reshape(1, -1)
        decoded = expected_level_to_continuous(levels, anchors)
        self.assertLess(decoded[0, 0], decoded[1, 0])

    def test_single_class_fallback_is_finite_and_deterministic(self):
        x = np.arange(48, dtype=float).reshape(16, 3)
        y = np.column_stack([np.zeros(16), np.linspace(-1, 1, 16)])
        kwargs = dict(max_leaf_nodes=7, min_samples_leaf=4,
                      learning_rate=0.05, l2_regularization=1.0,
                      max_iter=20, random_state=17)
        first = fit_predict_cumulative_hgb(x[:12], y[:12], x[12:], **kwargs)
        second = fit_predict_cumulative_hgb(x[:12], y[:12], x[12:], **kwargs)
        self.assertEqual(first.continuous.shape, (4, 2))
        self.assertTrue(np.isfinite(first.continuous).all())
        np.testing.assert_allclose(first.continuous, second.continuous)
~~~

- [x] **Step 2: Verify RED**

~~~bash
conda run -n mathorcup python -m unittest \
  '02_代码/Q1_感官评分/tests/test_ordered_models.py' -v
~~~

Expected: import failure for ordered_models.

- [x] **Step 3: Implement minimal ordered model**

Use this result type and primitives:

~~~python
@dataclass(frozen=True)
class OrdinalPrediction:
    continuous: np.ndarray
    expected_level: np.ndarray
    survival_probability: np.ndarray
    anchors: np.ndarray

def training_quantile_anchors(y_train, n_levels=7):
    return np.quantile(np.asarray(y_train, dtype=float),
                       np.linspace(0.0, 1.0, n_levels), axis=0).T

def project_survival_probabilities(probabilities):
    return np.minimum.accumulate(
        np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0), axis=-1)

def decode_expected_level(survival_probability):
    return np.asarray(survival_probability, dtype=float).sum(axis=-1)
~~~

fit_predict_cumulative_hgb must fit a fold-local median imputer with missing indicators, then six HistGradientBoostingClassifier instances per target. A single-class threshold returns its training mean probability. Decode expected levels with per-target np.interp over training anchors only.

- [x] **Step 4: Verify GREEN and the full Q1 suite**

~~~bash
conda run -n mathorcup python -m unittest \
  '02_代码/Q1_感官评分/tests/test_ordered_models.py' -v
conda run -n mathorcup python -m unittest discover \
  -s '02_代码/Q1_感官评分/tests' -p 'test_*.py' -v
~~~

- [x] **Step 5: Commit**

~~~bash
git add '02_代码/Q1_感官评分/src/ordered_models.py' \
  '02_代码/Q1_感官评分/tests/test_ordered_models.py'
git commit -m 'feat(q1): add leakage-safe cumulative ordinal HGB'
~~~

## Task 3: Implement continuous HGB and constrained blending

**Files:**
- Create: 02_代码/Q1_感官评分/src/ensemble_models.py
- Create: 02_代码/Q1_感官评分/tests/test_ensemble_models.py

- [x] **Step 1: Write failing tests**

~~~python
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from ensemble_models import (
    fit_predict_hgb_multioutput,
    select_feasible_convex_blend,
    shrink_target_weights,
    simplex_weight_grid,
)

class EnsembleModelTests(unittest.TestCase):
    def test_simplex_grid(self):
        weights = simplex_weight_grid(3, 0.5)
        self.assertTrue((weights >= 0).all())
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)

    def test_feasible_blend_keeps_continuous_quality(self):
        y = np.arange(20, dtype=float).reshape(10, 2)
        stack = np.stack([y.copy(), y[::-1].copy()], axis=-1)
        result = select_feasible_convex_blend(
            y, stack, simplex_weight_grid(2, 0.5),
            r2_floor=0.9, mae_ceiling=0.1)
        np.testing.assert_allclose(result.weights, [1.0, 0.0])

    def test_shrinkage_one_returns_global_weights(self):
        target = np.array([[1.0, 0.0], [0.0, 1.0]])
        global_weight = np.array([0.75, 0.25])
        shrunk = shrink_target_weights(target, global_weight, 1.0)
        np.testing.assert_allclose(shrunk, np.tile(global_weight, (2, 1)))
~~~

- [x] **Step 2: Verify RED**

~~~bash
conda run -n mathorcup python -m unittest \
  '02_代码/Q1_感官评分/tests/test_ensemble_models.py' -v
~~~

- [x] **Step 3: Implement exact public API**

~~~python
@dataclass(frozen=True)
class BlendSelection:
    weights: np.ndarray
    metrics: dict[str, float]
    feasible: bool

def simplex_weight_grid(n_models: int, step: float) -> np.ndarray:
    units = int(round(1.0 / step))
    if n_models < 1 or not np.isclose(units * step, 1.0):
        raise ValueError("step must divide one and n_models must be positive")
    rows = [row for row in product(range(units + 1), repeat=n_models)
            if sum(row) == units]
    return np.asarray(rows, dtype=float) / units

def apply_convex_blend(prediction_stack: np.ndarray,
                       weights: np.ndarray) -> np.ndarray:
    stack = np.asarray(prediction_stack, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if stack.ndim != 3 or stack.shape[-1] != len(weight):
        raise ValueError("prediction stack and weights disagree")
    if (weight < 0).any() or not np.isclose(weight.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    return np.tensordot(stack, weight, axes=([-1], [0]))

def select_feasible_convex_blend(y_true: np.ndarray,
                                 prediction_stack: np.ndarray,
                                 candidates: np.ndarray,
                                 r2_floor: float,
                                 mae_ceiling: float) -> BlendSelection:
    scored = []
    for order, weight in enumerate(candidates):
        metrics = matrix_metrics(y_true, apply_convex_blend(prediction_stack, weight))
        feasible = metrics["r2"] >= r2_floor and metrics["mae"] <= mae_ceiling
        violation = max(0.0, r2_floor - metrics["r2"]) + max(
            0.0, metrics["mae"] - mae_ceiling)
        rank = (metrics["spearman"], metrics["qwk_proxy"],
                metrics["acc_pm1_proxy"], metrics["r2"], -metrics["mae"])
        scored.append((feasible, violation, rank, -order, weight, metrics))
    feasible_rows = [row for row in scored if row[0]]
    chosen = max(feasible_rows, key=lambda row: (row[2], row[3])) if feasible_rows \
        else min(scored, key=lambda row: (row[1], -row[2][0], -row[3]))
    return BlendSelection(np.asarray(chosen[4]), chosen[5], bool(chosen[0]))

def shrink_target_weights(target_weights: np.ndarray,
                          global_weights: np.ndarray,
                          shrinkage: float) -> np.ndarray:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    return ((1.0 - shrinkage) * np.asarray(target_weights, dtype=float)
            + shrinkage * np.asarray(global_weights, dtype=float))

def fit_predict_hgb_multioutput(x_train: np.ndarray, y_train: np.ndarray,
                                x_validation: np.ndarray,
                                max_leaf_nodes: int,
                                min_samples_leaf: int,
                                learning_rate: float,
                                l2_regularization: float,
                                max_iter: int,
                                random_state: int) -> np.ndarray:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train = imputer.fit_transform(x_train)
    validation = imputer.transform(x_validation)
    columns = []
    for position in range(np.asarray(y_train).shape[1]):
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            max_iter=max_iter,
            random_state=random_state + position)
        columns.append(model.fit(train, y_train[:, position]).predict(validation))
    return np.column_stack(columns)
~~~

Feasible blends are sorted by descending (Spearman, QWK, ACC±1, R², -MAE). If none pass the continuous floors, return the smallest normalized R²/MAE violation with feasible=False. Preserve input order for exact ties.

- [x] **Step 4: Verify GREEN and commit**

~~~bash
conda run -n mathorcup python -m unittest discover \
  -s '02_代码/Q1_感官评分/tests' -p 'test_*.py' -v
git add '02_代码/Q1_感官评分/src/ensemble_models.py' \
  '02_代码/Q1_感官评分/tests/test_ensemble_models.py'
git commit -m 'feat(q1): add constrained continuous ordinal blending'
~~~

## Task 4: Build the shared nested-OOF runner

**Files:**
- Create: 02_代码/Q1_感官评分/src/second_round_runner.py
- Create: 02_代码/Q1_感官评分/tests/test_second_round_runner.py

- [x] **Step 1: Write failing split and completeness tests**

~~~python
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from second_round_runner import assert_complete_oof, outer_indices, parse_repeats

class SecondRoundRunnerTests(unittest.TestCase):
    def test_outer_indices_are_disjoint_and_complete(self):
        splits = pd.DataFrame({
            "repeat": [0, 0], "fold": [0, 0], "row_position": [1, 3]})
        train, validation = outer_indices(splits, 0, 0, 5)
        self.assertEqual(set(train) & set(validation), set())
        self.assertEqual(set(train) | set(validation), set(range(5)))

    def test_incomplete_oof_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            assert_complete_oof(np.full((2, 5, 3), np.nan))

    def test_repeat_parser_is_sorted_and_bounded(self):
        self.assertEqual(parse_repeats("2,0,1", list(range(10))), [0, 1, 2])
        with self.assertRaises(ValueError):
            parse_repeats("10", list(range(10)))
~~~

- [x] **Step 2: Verify RED**

~~~bash
conda run -n mathorcup python -m unittest \
  '02_代码/Q1_感官评分/tests/test_second_round_runner.py' -v
~~~

- [x] **Step 3: Implement split and artifact functions**

~~~python
def parse_repeats(text: str, available: Sequence[int]) -> list[int]:
    selected = sorted({int(value) for value in text.split(",") if value.strip()})
    if not selected or not set(selected).issubset(set(available)):
        raise ValueError("requested repeats are not available")
    return selected

def outer_indices(splits: pd.DataFrame, repeat: int, fold: int,
                  n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    validation = splits.loc[
        splits["repeat"].eq(repeat) & splits["fold"].eq(fold),
        "row_position"].to_numpy(int)
    mask = np.zeros(n_samples, dtype=bool)
    mask[validation] = True
    return np.flatnonzero(~mask), validation

def make_inner_splits(n_samples: int, n_splits: int,
                      random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=n_splits, shuffle=True,
                     random_state=random_state)
    return list(splitter.split(np.arange(n_samples)))

def assert_complete_oof(prediction: np.ndarray) -> None:
    if not np.isfinite(np.asarray(prediction, dtype=float)).all():
        raise RuntimeError("incomplete or non-finite OOF predictions")

def save_prediction_archive(path: Path, *, y_true: np.ndarray,
                            beer: np.ndarray, style: np.ndarray,
                            target_code: np.ndarray, repeats: np.ndarray,
                            fold_assignment: np.ndarray,
                            predictions: Mapping[str, np.ndarray]) -> None:
    for value in predictions.values():
        assert_complete_oof(value)
    np.savez_compressed(
        path, y_true=y_true, beer=beer, style=style,
        target_code=target_code, repeat=repeats,
        fold_assignment=fold_assignment, **predictions)

def write_metric_artifacts(experiment_id: str, archive_path: Path,
                           prediction_keys: Sequence[str],
                           suffix: str) -> None:
    archive = np.load(archive_path, allow_pickle=True)
    y_true = np.asarray(archive["y_true"], dtype=float)
    targets = np.asarray(archive["target_code"])
    frames = []
    for key in prediction_keys:
        prediction = np.asarray(archive[key], dtype=float)
        for position, repeat in enumerate(archive["repeat"]):
            frames.append(evaluate_prediction_matrix(
                y_true, prediction[position], targets, experiment_id,
                f"{key}:repeat_{int(repeat)}"))
        frames.append(evaluate_prediction_matrix(
            y_true, prediction.mean(axis=0), targets, experiment_id,
            f"{key}:mean_oof"))
    by_target = pd.concat(frames, ignore_index=True)
    stem = f"_{suffix}" if suffix else ""
    by_target.to_csv(
        RESULT_DIR / f"q1_extended_metrics_by_target_{experiment_id}{stem}.csv",
        index=False)
    summarise_target_metrics(by_target).to_csv(
        RESULT_DIR / f"q1_extended_metrics_{experiment_id}{stem}.csv",
        index=False)
~~~

Screening suffixes must be preserved, so partial output cannot overwrite full output.

- [x] **Step 4: Verify GREEN and commit**

~~~bash
conda run -n mathorcup python -m unittest discover \
  -s '02_代码/Q1_感官评分/tests' -p 'test_*.py' -v
git add '02_代码/Q1_感官评分/src/second_round_runner.py' \
  '02_代码/Q1_感官评分/tests/test_second_round_runner.py'
git commit -m 'feat(q1): add frozen nested OOF runner contract'
~~~

## Task 5: Run E004 cumulative-ordinal screening

**Files:**
- Create: 02_代码/Q1_感官评分/configs/E004_ordinal_hgb.yaml
- Create: 02_代码/Q1_感官评分/src/run_e004_ordinal_hgb.py
- Modify: 02_代码/Q1_感官评分/configs/README.md
- Modify: docs/experiment_log.md

- [x] **Step 1: Add a failing config contract test**

~~~python
def test_e004_config(self):
    import yaml
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(
        (root / "02_代码/Q1_感官评分/configs/E004_ordinal_hgb.yaml").read_text())
    self.assertEqual(config["screening_repeats"], [0, 1, 2])
    self.assertEqual(len(config["profiles"]), 3)
~~~

- [x] **Step 2: Verify RED, then create the frozen config**

~~~yaml
experiment_id: E004
screening_repeats: [0, 1, 2]
full_repeats: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
inner_splits: 3
random_state: 20261023
profiles:
  - {name: compact, max_leaf_nodes: 7, min_samples_leaf: 20, learning_rate: 0.05, l2_regularization: 1.0, max_iter: 80}
  - {name: balanced, max_leaf_nodes: 15, min_samples_leaf: 20, learning_rate: 0.05, l2_regularization: 1.0, max_iter: 80}
  - {name: regularized, max_leaf_nodes: 7, min_samples_leaf: 10, learning_rate: 0.05, l2_regularization: 10.0, max_iter: 80}
selection_metric: macro_mae_then_ordered
~~~

- [x] **Step 3: Implement --stage screening and --stage full**

For each outer fold, select one shared profile using inner OOF, fit on the entire outer-training fold, and predict the untouched outer-validation fold. Screening writes only files with _screening suffix.

- [x] **Step 4: Run tests and the nine-fold screening**

~~~bash
conda run -n mathorcup python -m unittest discover \
  -s '02_代码/Q1_感官评分/tests' -p 'test_*.py' -v
conda run -n mathorcup python \
  '02_代码/Q1_感官评分/src/run_e004_ordinal_hgb.py' --stage screening
~~~

Expected: finite OOF predictions, selected-profile CSV, metrics and manifest for repeats 0–2.

- [x] **Step 5: Record and commit**

~~~bash
git add '02_代码/Q1_感官评分/configs/E004_ordinal_hgb.yaml' \
  '02_代码/Q1_感官评分/src/run_e004_ordinal_hgb.py' \
  '02_代码/Q1_感官评分/configs/README.md' \
  '02_代码/Q1_感官评分/tests/test_second_round_runner.py' \
  '03_结果/Q1_感官评分/' docs/experiment_log.md
git commit -m 'feat(q1): complete E004 ordinal HGB screening'
~~~

## Task 6: Run E005 continuous-ensemble screening

**Files:**
- Create: 02_代码/Q1_感官评分/configs/E005_continuous_ensemble.yaml
- Create: 02_代码/Q1_感官评分/src/run_e005_continuous_ensemble.py
- Modify: docs/experiment_log.md

- [x] **Step 1: Add a failing config test**

~~~python
def test_e005_config(self):
    import yaml
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(
        (root / "02_代码/Q1_感官评分/configs/E005_continuous_ensemble.yaml").read_text())
    self.assertEqual(config["base_models"],
                     ["extra_trees", "hgb", "ridge_quantile", "pls2_quantile"])
    self.assertEqual(config["weight_step"], 0.1)
~~~

- [x] **Step 2: Verify RED, then freeze the config**

~~~yaml
experiment_id: E005
screening_repeats: [0, 1, 2]
full_repeats: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
inner_splits: 3
random_state: 20261023
base_models: [extra_trees, hgb, ridge_quantile, pls2_quantile]
weight_step: 0.1
continuous_tolerance: {r2_drop: 0.002, mae_increase: 0.001}
hgb_profiles:
  - {name: compact, max_leaf_nodes: 7, min_samples_leaf: 20, learning_rate: 0.05, l2_regularization: 1.0, max_iter: 100}
  - {name: balanced, max_leaf_nodes: 15, min_samples_leaf: 20, learning_rate: 0.05, l2_regularization: 1.0, max_iter: 100}
~~~

- [x] **Step 3: Implement nested base predictions and global convex weights**

Inside each outer fold, create inner OOF predictions for all four bases, choose base parameters by inner macro MAE, select weights from the 0.1 simplex, refit on all outer-training samples, and predict outer validation. Save each base and pred_continuous_blend.

- [x] **Step 4: Run screening**

~~~bash
conda run -n mathorcup python \
  '02_代码/Q1_感官评分/src/run_e005_continuous_ensemble.py' --stage screening
~~~

Expected: nine complete folds and weights that are nonnegative and sum to one.

- [x] **Step 5: Record and commit**

~~~bash
git add '02_代码/Q1_感官评分/configs/E005_continuous_ensemble.yaml' \
  '02_代码/Q1_感官评分/src/run_e005_continuous_ensemble.py' \
  '02_代码/Q1_感官评分/tests/test_second_round_runner.py' \
  '03_结果/Q1_感官评分/' docs/experiment_log.md
git commit -m 'feat(q1): complete E005 continuous ensemble screening'
~~~

## Task 7: Run E006 dual-route screening and promotion gate

**Files:**
- Create: 02_代码/Q1_感官评分/configs/E006_dual_route.yaml
- Create: 02_代码/Q1_感官评分/src/run_e006_dual_route.py
- Modify: docs/experiment_log.md

- [ ] **Step 1: Add a failing hard-threshold test**

~~~python
def test_e006_thresholds(self):
    import yaml
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(
        (root / "02_代码/Q1_感官评分/configs/E006_dual_route.yaml").read_text())
    self.assertEqual(config["hard_thresholds"], {
        "r2_min": 0.2359, "mae_max": 0.3422,
        "spearman_min": 0.4260, "qwk_proxy_min": 0.4026,
        "acc_pm1_proxy_min": 0.5734})
~~~

- [ ] **Step 2: Verify RED and create config**

Freeze ordinal shares 0.00 through 0.50 in increments of 0.05, the five thresholds above, and the same screening/full repeat IDs.

- [ ] **Step 3: Implement inner-only route blending**

E006 must regenerate selected E004/E005 inner predictions inside every outer-training fold. It must not learn weights from top-level screening OOF files. Reject weights outside the inner continuous tolerance, then rank feasible weights by Spearman, QWK, ACC±1, R², and -MAE.

- [ ] **Step 4: Run screening and gate full execution**

~~~bash
conda run -n mathorcup python \
  '02_代码/Q1_感官评分/src/run_e006_dual_route.py' --stage screening
~~~

Promote if four of five screening means improve over E003, including Spearman or ACC±1, while R² drops by at most 0.002 and MAE rises by at most 0.001. If promoted:

~~~bash
conda run -n mathorcup python \
  '02_代码/Q1_感官评分/src/run_e006_dual_route.py' --stage full
~~~

- [ ] **Step 5: Commit the decision**

~~~bash
git add '02_代码/Q1_感官评分/configs/E006_dual_route.yaml' \
  '02_代码/Q1_感官评分/src/run_e006_dual_route.py' \
  '02_代码/Q1_感官评分/tests/test_second_round_runner.py' \
  '03_结果/Q1_感官评分/' docs/experiment_log.md
git commit -m 'feat(q1): evaluate E006 dual route promotion gate'
~~~

## Task 8: Apply E007 shrinkage gating only when required

**Files:**
- Create: 02_代码/Q1_感官评分/configs/E007_shrunk_gating.yaml
- Create: 02_代码/Q1_感官评分/src/run_e007_shrunk_gating.py
- Modify: docs/experiment_log.md

- [ ] **Step 1: Add a failing config test**

~~~python
def test_e007_config(self):
    import yaml
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(
        (root / "02_代码/Q1_感官评分/configs/E007_shrunk_gating.yaml").read_text())
    self.assertEqual(config["shrinkage_strengths"],
                     [0.0, 0.25, 0.5, 0.75, 1.0])
    self.assertEqual(config["entry_condition"],
                     "E006_misses_at_least_one_hard_threshold")
~~~

- [ ] **Step 2: Verify RED and create config**

Use the exact values in the test, the same repeat IDs, and the E006 route-weight grid.

- [ ] **Step 3: Implement one shared shrinkage strength**

For every target, find its inner-OOF best route weight, then compute:

~~~python
final_weight = shrinkage * global_weight + (1.0 - shrinkage) * target_weight
~~~

Choose one shared shrinkage strength using inner macro metrics. If E006 passes all hard thresholds, write a SKIPPED manifest and fit nothing.

- [ ] **Step 4: Run screening, then full only if promoted**

~~~bash
conda run -n mathorcup python \
  '02_代码/Q1_感官评分/src/run_e007_shrunk_gating.py' --stage screening
~~~

Apply the E006 screening rule unchanged.

- [ ] **Step 5: Commit E007 or its SKIPPED manifest**

~~~bash
git add '02_代码/Q1_感官评分/configs/E007_shrunk_gating.yaml' \
  '02_代码/Q1_感官评分/src/run_e007_shrunk_gating.py' \
  '02_代码/Q1_感官评分/tests/test_second_round_runner.py' \
  '03_结果/Q1_感官评分/' docs/experiment_log.md
git commit -m 'feat(q1): resolve E007 shrinkage gating decision'
~~~

## Task 9: Audit the winner and hand off

**Files:**
- Create: 02_代码/Q1_感官评分/src/audit_q1_second_round.py
- Create: docs/analysis/2026初赛/Q1_二轮优化报告.md
- Modify: 02_代码/Q1_感官评分/README.md
- Modify: 02_代码/Q1_感官评分/src/README.md
- Modify: docs/analysis/2026初赛/BZD与agent对照.md
- Modify: docs/experiment_log.md
- Modify: docs/collab/board.md
- Modify: docs/collab/to_zcode.md
- Modify: task_plan.md
- Modify: progress.md
- Modify: findings.md

- [ ] **Step 1: Write the failing promotion test**

~~~python
def test_promotion_requires_all_metrics_and_seven_repeats(self):
    from audit_q1_second_round import promotion_decision
    means = {"r2": 0.24, "mae": 0.34, "spearman": 0.43,
             "qwk_proxy": 0.41, "acc_pm1_proxy": 0.58}
    counts = {name: 7 for name in means}
    self.assertTrue(promotion_decision(means, counts)["promoted"])
    counts["acc_pm1_proxy"] = 6
    self.assertFalse(promotion_decision(means, counts)["promoted"])
~~~

- [ ] **Step 2: Implement audit_q1_second_round.py**

It must compare every full candidate with E003 by repeat; run 20,000 paired beer-level bootstrap resamples for MAE and absolute rank error; report bottom-quartile target performance and worst-style MAE; compute five mean-threshold and five 7/10 flags; and write q1_second_round_decision.json, q1_second_round_comparison.csv, and q1_second_round_bootstrap.csv.

- [ ] **Step 3: Run verification**

~~~bash
conda run -n mathorcup python -m unittest discover \
  -s '02_代码/Q1_感官评分/tests' -p 'test_*.py' -v
conda run -n mathorcup python \
  '02_代码/Q1_感官评分/src/audit_q1_second_round.py'
git diff --check
~~~

Expected: tests pass, every decision flag is explicit, and E001–E003 plus final_v3_draft remain unchanged.

- [ ] **Step 4: Write the optimization report**

The report must include the frozen task, leakage controls, E003/BZD/E004–E007 table, screening/full separation, paired-repeat and tail evidence, accepted or rejected decision, exact rejection reasons, and whether E008 is justified. Every number must cite a machine-generated result path.

- [ ] **Step 5: Commit and push the decision checkpoint**

~~~bash
git add '02_代码/Q1_感官评分/' '03_结果/Q1_感官评分/' \
  'docs/analysis/2026初赛/Q1_二轮优化报告.md' \
  'docs/analysis/2026初赛/BZD与agent对照.md' \
  docs/experiment_log.md docs/collab/board.md docs/collab/to_zcode.md \
  task_plan.md progress.md findings.md
git commit -m 'collab(Q1-OPT): complete Q1 second-round multi-metric audit'
git push origin main
~~~

- [ ] **Step 6: Verify handoff state**

~~~bash
git status --porcelain
git rev-parse HEAD
git rev-parse origin/main
~~~

Expected: empty status and identical hashes. If promoted, stop before V4 figures or Word so the user can review the numerical checkpoint.

## Plan self-review

- Spec coverage: Tasks 1–9 cover metric reuse, ordinal prediction, continuous ensemble, nested OOF, E004–E007 sequencing, hard gates, tail/bootstrap checks, artifacts, report and collaboration handoff.
- Completeness scan: profiles, thresholds, repeats, weight grids, shrinkage values, commands, file paths and expected outcomes are explicit.
- Type consistency: matrix predictions are (samples, targets), base stacks are (samples, targets, models), ordinal probabilities are (samples, targets, 6), and OOF tensors are (repeats, samples, targets).
- Scope control: E008 is deferred until the approved stop rule justifies a separate dependency and resource design.

