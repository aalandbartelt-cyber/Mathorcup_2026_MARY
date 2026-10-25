# Q2 Evidence-First Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Q2 E112 only if a fold-safe partial-correlation graph, hierarchical conditional attribution, and error-controlled stability selection produce a demonstrably stronger 50×3 key-compound evidence chain.

**Architecture:** Keep E105 ExtraTrees as the frozen predictive baseline while adding three independent evidence families: nonlinear conditional prediction, shrinkage partial effects, and stable graph modules. Run E113—E117 as isolated candidates, fuse only candidates that pass leakage and stability audits in E118, then append chemistry post hoc in E119 and generate versioned R/ggplot2 figures.

**Tech Stack:** Python 3.10, NumPy 2.2, pandas 2.3, SciPy 1.15, scikit-learn 1.7, NetworkX, PyYAML, unittest, R/ggplot2/patchwork, CPU-first execution.

---

## File map

| Path | Responsibility |
|---|---|
| `02_代码/Q2_关键化合物/src/partial_graph.py` | Fold-local rank-Gaussian transform, Graphical Lasso, partial correlations, Louvain modules and graph stability |
| `02_代码/Q2_关键化合物/src/hierarchical_importance.py` | Module joint permutation and within-module conditional residual permutation |
| `02_代码/Q2_关键化合物/src/target_partial_effects.py` | Ledoit–Wolf target partial correlations and grouped multi-task coefficient rankings |
| `02_代码/Q2_关键化合物/src/complementary_stability.py` | Style-stratified complementary halves, selection probabilities, null controls and empirical FDP |
| `02_代码/Q2_关键化合物/src/evidence_consensus.py` | Three-family rank fusion, stability/style gates and replacement decision |
| `02_代码/Q2_关键化合物/src/run_e113_partial_graph.py` | E113 graph comparison and fold-local module artifacts |
| `02_代码/Q2_关键化合物/src/run_e114_partial_effects.py` | E114 fold-local shrinkage partial effects |
| `02_代码/Q2_关键化合物/src/run_e115_hierarchical_importance.py` | E115 hierarchical predictive attribution |
| `02_代码/Q2_关键化合物/src/run_e116_grouped_multitask.py` | E116 limited target-sharing ablation |
| `02_代码/Q2_关键化合物/src/run_e117_stability_controls.py` | E117 CPSS and null-control execution |
| `02_代码/Q2_关键化合物/src/run_e118_evidence_consensus.py` | E118 final statistical ranking and replacement gate |
| `02_代码/Q2_关键化合物/src/run_e119_chemical_audit.py` | E119 post-ranking chemical evidence audit |
| `02_代码/Q2_关键化合物/src/plot_q2_v2.R` | Four versioned V2 main figures and source-data manifests |
| `02_代码/Q2_关键化合物/tests/test_partial_graph.py` | Graph construction and leakage-boundary invariants |
| `02_代码/Q2_关键化合物/tests/test_hierarchical_importance.py` | Module and conditional-permutation invariants |
| `02_代码/Q2_关键化合物/tests/test_target_partial_effects.py` | Partial-effect and grouped-target tests |
| `02_代码/Q2_关键化合物/tests/test_complementary_stability.py` | Complementarity, frequency and null-control tests |
| `02_代码/Q2_关键化合物/tests/test_evidence_consensus.py` | Fusion, gates and complete 50×3 contract |
| `docs/analysis/2026初赛/Q2_证据链优化报告.md` | E112 versus E113—E119 decision report |

Execution must begin in a dedicated worktree created from current `main`; do not run numerical experiments in the Q1 optimization worktree.

## Task 1: Implement fold-local sparse partial graphs

**Files:**
- Create: `02_代码/Q2_关键化合物/src/partial_graph.py`
- Create: `02_代码/Q2_关键化合物/tests/test_partial_graph.py`
- Create: `02_代码/Q2_关键化合物/configs/E113_partial_graph.yaml`
- Modify: `02_代码/Q2_关键化合物/configs/README.md`

- [x] **Step 1: Write failing graph tests**

```python
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from partial_graph import (
    FoldGaussianizer, precision_to_partial_correlation,
    stable_edge_table, coclustering_matrix,
)

class PartialGraphTests(unittest.TestCase):
    def test_gaussianizer_does_not_refit_on_validation(self):
        train = np.arange(30, dtype=float).reshape(10, 3)
        validation = np.full((2, 3), 999.0)
        transformer = FoldGaussianizer().fit(train)
        before = transformer.quantiles_.copy()
        transformed = transformer.transform(validation)
        np.testing.assert_array_equal(before, transformer.quantiles_)
        self.assertTrue(np.isfinite(transformed).all())

    def test_partial_correlation_is_symmetric_with_unit_diagonal(self):
        precision = np.array([[2.0, -0.5], [-0.5, 3.0]])
        partial = precision_to_partial_correlation(precision)
        np.testing.assert_allclose(partial, partial.T)
        np.testing.assert_allclose(np.diag(partial), 1.0)

    def test_coclustering_is_label_invariant(self):
        first = np.array([0, 0, 1, 1])
        relabelled = np.array([7, 7, 3, 3])
        np.testing.assert_allclose(
            coclustering_matrix([first]), coclustering_matrix([relabelled]))
```

- [x] **Step 2: Verify RED**

Run:

```bash
conda run -n mathorcup python -m unittest \
  '02_代码/Q2_关键化合物/tests/test_partial_graph.py' -v
```

Expected: import failure for `partial_graph`.

- [x] **Step 3: Implement the graph API**

Use a training-only quantile transformer and explicit precision conversion:

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

class FoldGaussianizer:
    def fit(self, x):
        values = np.asarray(x, dtype=float)
        self.model_ = QuantileTransformer(
            n_quantiles=min(100, len(values)), output_distribution="normal",
            subsample=None, random_state=20261025,
        ).fit(values)
        self.quantiles_ = self.model_.quantiles_.copy()
        return self
    def transform(self, x):
        return self.model_.transform(np.asarray(x, dtype=float))
    def fit_transform(self, x):
        return self.fit(x).transform(x)

def precision_to_partial_correlation(precision):
    matrix = np.asarray(precision, dtype=float)
    scale = np.sqrt(np.clip(np.diag(matrix), 1e-12, None))
    partial = -matrix / np.outer(scale, scale)
    np.fill_diagonal(partial, 1.0)
    return (partial + partial.T) / 2.0

def coclustering_matrix(label_vectors):
    labels = [np.asarray(value) for value in label_vectors]
    if not labels:
        raise ValueError("at least one module assignment is required")
    result = np.zeros((len(labels[0]), len(labels[0])), dtype=float)
    for vector in labels:
        if len(vector) != len(labels[0]):
            raise ValueError("module assignments must have equal length")
        result += vector[:, None] == vector[None, :]
    return result / len(labels)
```

Add `fit_graphical_lasso`, `louvain_membership`, `edge_table`, `stable_edge_table`, held-out log-likelihood and graph-structure summaries. Enforce symmetric finite matrices and deterministic node order. Louvain uses `seed=20261025`; zero-edge nodes receive unique singleton labels.

Freeze this config:

```yaml
experiment: E113
question: Q2
candidate_scope: strict_compounds_218
outer_cv_file: 03_结果/Q1_感官评分/q1_cv_splits.csv
inner_splits: 3
alpha_grid: [0.02, 0.05, 0.10, 0.20, 0.35]
selection_rule: one_se_loglik_then_max_edge_stability_then_larger_alpha
partial_edge_threshold: 0.000001
module_algorithm: louvain
module_seed: 20261025
random_state: 20261025
gpu_used: false
```

- [x] **Step 4: Verify GREEN**

Run the new test and all existing Q2 tests. Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add '02_代码/Q2_关键化合物/src/partial_graph.py' \
  '02_代码/Q2_关键化合物/tests/test_partial_graph.py' \
  '02_代码/Q2_关键化合物/configs/E113_partial_graph.yaml' \
  '02_代码/Q2_关键化合物/configs/README.md'
git commit -m 'feat(q2): add fold-local sparse partial graph primitives'
```

## Task 2: Run E113 graph and module comparison

**Files:**
- Create: `02_代码/Q2_关键化合物/src/run_e113_partial_graph.py`
- Modify: `02_代码/Q2_关键化合物/tests/test_partial_graph.py`
- Create: `03_结果/Q2_关键化合物/q2_graph_fold_summary_E113.csv`
- Create: `03_结果/Q2_关键化合物/q2_partial_edges_E113.csv.gz`
- Create: `03_结果/Q2_关键化合物/q2_module_membership_E113.csv.gz`
- Create: `03_结果/Q2_关键化合物/q2_coclustering_E113.csv.gz`
- Create: `03_结果/Q2_关键化合物/q2_run_manifest_E113.json`

- [x] **Step 1: Add the selection-rule test**

```python
def test_one_se_rule_prefers_stable_sparser_graph(self):
    rows = pd.DataFrame({
        "alpha": [0.05, 0.10, 0.20],
        "mean_loglik": [-10.0, -10.1, -10.5],
        "se_loglik": [0.2, 0.2, 0.2],
        "mean_edge_jaccard": [0.40, 0.60, 0.70],
    })
    chosen = select_graph_alpha(rows)
    self.assertEqual(chosen, 0.10)
```

- [x] **Step 2: Verify RED, then implement the runner**

For each of the 30 frozen outer folds:

1. load the strict 218 fields from `q2_candidate_dictionary.csv`;
2. fit imputation and `FoldGaussianizer` on outer training only;
3. use three inner folds to score every alpha by validation log-likelihood;
4. compute inner edge Jaccard for each alpha;
5. select by the frozen 1-SE rule;
6. refit the graph on the whole outer-training fold;
7. save every non-zero signed partial edge and every module membership.

After 30 folds, compute edge selection frequency, sign consistency and the 218×218 co-clustering matrix. Write the all-data descriptive graph separately with `scope=full_data_descriptive`; never merge it into fold rows.

- [ ] **Step 3: Run E113 and audit outputs**

```bash
conda run -n mathorcup python \
  '02_代码/Q2_关键化合物/src/run_e113_partial_graph.py'
```

Expected: 30 fold summaries; every fold contains one selected alpha; full-data rows are explicitly marked descriptive; no validation row is used while fitting its graph.

- [ ] **Step 4: Compare graph candidates**

Write `q2_graph_decision_E113.json` with:

```json
{
  "promote_partial_graph": true,
  "criteria": {
    "heldout_loglik_finite": true,
    "edge_or_module_stability_improves": true,
    "no_full_graph_in_validation": true
  }
}
```

Set `promote_partial_graph` from machine values, not by hand. If false, retain the E107 stable Spearman modules and continue E114 as a conditional-effect-only ablation.

- [ ] **Step 5: Commit**

```bash
git add '02_代码/Q2_关键化合物/src/run_e113_partial_graph.py' \
  '02_代码/Q2_关键化合物/tests/test_partial_graph.py' \
  '03_结果/Q2_关键化合物/' docs/experiment_log.md
git commit -m 'exp(q2): evaluate E113 fold-local partial graph'
```

## Task 3: Implement and run E114 shrinkage partial effects

**Files:**
- Create: `02_代码/Q2_关键化合物/src/target_partial_effects.py`
- Create: `02_代码/Q2_关键化合物/tests/test_target_partial_effects.py`
- Create: `02_代码/Q2_关键化合物/configs/E114_partial_effects.yaml`
- Create: `02_代码/Q2_关键化合物/src/run_e114_partial_effects.py`

- [x] **Step 1: Write failing partial-effect tests**

```python
from target_partial_effects import ledoit_wolf_target_partial, rank_signed_effects

def test_direct_signal_has_largest_partial_effect(self):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(160, 5))
    y = 2.0 * x[:, 2] + rng.normal(scale=0.2, size=160)
    effect = ledoit_wolf_target_partial(x, y)
    self.assertEqual(int(np.argmax(np.abs(effect))), 2)

def test_signed_rank_is_deterministic(self):
    effect = np.array([0.5, -0.5, 0.1])
    first = rank_signed_effects(effect, ["b", "a", "c"])
    second = rank_signed_effects(effect, ["b", "a", "c"])
    pd.testing.assert_frame_equal(first, second)
```

- [x] **Step 2: Implement the exact estimator**

```python
from sklearn.covariance import LedoitWolf

def ledoit_wolf_target_partial(x, y):
    matrix = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    precision = LedoitWolf().fit(matrix).precision_
    scale = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    return -precision[:-1, -1] / (scale[:-1] * scale[-1])
```

Use fold-local median imputation, missing indicators only in prediction models, and the training-fold Gaussianizer for partial effects. Rank by absolute effect; preserve the sign separately. Aggregate 30 outer-training rankings by median rank, Top-3 frequency, Top-10 frequency and sign consistency.

Freeze config:

```yaml
experiment: E114
question: Q2
outer_folds: frozen_10x3
estimator: ledoit_wolf_precision_of_rank_gaussian_X_plus_standardized_y
candidate_scope: strict_compounds_218
aggregation: [median_rank, top3_frequency, top10_frequency, sign_consistency]
gpu_used: false
```

- [ ] **Step 3: Run tests and E114**

Expected outputs:

- `q2_fold_partial_effects_E114.csv.gz`, 30×50×218 rows;
- `q2_partial_effect_rankings_E114.csv`, 50×218 rows;
- `q2_anchor_recovery_E114.csv`;
- `q2_run_manifest_E114.json`.

- [ ] **Step 4: Commit**

```bash
git add '02_代码/Q2_关键化合物/src/target_partial_effects.py' \
  '02_代码/Q2_关键化合物/src/run_e114_partial_effects.py' \
  '02_代码/Q2_关键化合物/configs/E114_partial_effects.yaml' \
  '02_代码/Q2_关键化合物/tests/test_target_partial_effects.py' \
  '03_结果/Q2_关键化合物/' docs/experiment_log.md
git commit -m 'exp(q2): add E114 fold-local shrinkage partial effects'
```

## Task 4: Implement E115 hierarchical predictive attribution

**Files:**
- Create: `02_代码/Q2_关键化合物/src/hierarchical_importance.py`
- Create: `02_代码/Q2_关键化合物/tests/test_hierarchical_importance.py`
- Create: `02_代码/Q2_关键化合物/configs/E115_hierarchical_importance.yaml`
- Create: `02_代码/Q2_关键化合物/src/run_e115_hierarchical_importance.py`

- [ ] **Step 1: Write failing permutation tests**

```python
from hierarchical_importance import joint_permutation, residual_permutation

def test_joint_permutation_keeps_module_columns_together(self):
    x = np.arange(60, dtype=float).reshape(12, 5)
    changed, order = joint_permutation(x, [1, 3], np.random.default_rng(4))
    np.testing.assert_array_equal(changed[:, 1], x[order, 1])
    np.testing.assert_array_equal(changed[:, 3], x[order, 3])
    np.testing.assert_array_equal(changed[:, 0], x[:, 0])

def test_residual_permutation_preserves_conditional_mean(self):
    mean = np.linspace(0, 1, 20)
    residual = np.arange(20, dtype=float)
    changed = residual_permutation(mean, residual, np.random.default_rng(5))
    np.testing.assert_allclose(np.sort(changed - mean), np.sort(residual))
```

- [ ] **Step 2: Implement module and residual importance**

Expose these functions:

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def joint_permutation(x, columns, rng):
    values = np.asarray(x, dtype=float)
    selected = np.asarray(columns, dtype=int)
    order = rng.permutation(len(values))
    changed = values.copy()
    changed[:, selected] = values[order][:, selected]
    return changed, order

@dataclass
class ConditionalFeatureModel:
    pipeline: Pipeline
    n_conditioners: int

    def predict(self, conditioners, style):
        numeric = np.asarray(conditioners, dtype=float)
        frame = pd.DataFrame(numeric, columns=[
            f"x_{position}" for position in range(self.n_conditioners)
        ])
        frame["style"] = np.asarray(style, dtype=str)
        return np.asarray(self.pipeline.predict(frame), dtype=float)

def fit_conditional_feature_model(conditioners, feature, style):
    numeric = np.asarray(conditioners, dtype=float)
    frame = pd.DataFrame(numeric, columns=[
        f"x_{position}" for position in range(numeric.shape[1])
    ])
    frame["style"] = np.asarray(style, dtype=str)
    numeric_columns = list(frame.columns[:-1])
    transformer = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_columns),
        ("style", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["style"]),
    ])
    pipeline = Pipeline([
        ("transform", transformer),
        ("model", HistGradientBoostingRegressor(
            max_leaf_nodes=7, min_samples_leaf=20, learning_rate=0.05,
            l2_regularization=1.0, max_iter=100, random_state=20261025)),
    ])
    pipeline.fit(frame, np.asarray(feature, dtype=float))
    return ConditionalFeatureModel(pipeline, numeric.shape[1])

def residual_permutation(conditional_mean, residual, rng):
    mean = np.asarray(conditional_mean, dtype=float)
    error = np.asarray(residual, dtype=float)
    return mean + error[rng.permutation(len(error))]

def delta_mae(y_true, baseline_prediction, changed_prediction):
    truth = np.asarray(y_true, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    changed = np.asarray(changed_prediction, dtype=float)
    return float(np.mean(np.abs(truth - changed)) -
                 np.mean(np.abs(truth - baseline)))

def aggregate_hierarchical_importance(fold_rows):
    frame = pd.DataFrame(fold_rows).copy()
    grouped = frame.groupby(["target_code", "feature_code"], sort=False)
    result = grouped.agg(
        median_delta_mae=("delta_mae", "median"),
        positive_fraction=("delta_mae", lambda value: float((value > 0).mean())),
        fold_top3_frequency=("fold_rank", lambda value: float((value <= 3).mean())),
        median_fold_rank=("fold_rank", "median"),
    ).reset_index()
    result = result.sort_values(
        ["target_code", "positive_fraction", "median_delta_mae",
         "median_fold_rank", "feature_code"],
        ascending=[True, False, False, True, True], kind="mergesort")
    result["hierarchical_rank"] = result.groupby("target_code").cumcount() + 1
    return result
```

`fit_conditional_feature_model` uses `HistGradientBoostingRegressor` with the frozen compact profile and one-hot style indicators fitted on training rows. A singleton module uses ordinary permutation. Only fields from modules with positive median joint ΔMAE enter field-level residual permutation.

Freeze config:

```yaml
experiment: E115
prediction_backbones: [E105_ExtraTrees]
module_source: E113_if_promoted_else_E107
module_repeats_per_fold: 10
field_repeats_per_fold: 10
module_gate: {median_delta_mae_min: 0.0, positive_fraction_min: 0.60}
conditional_model: compact_hgb
candidate_scope: strict_compounds_218
random_state: 20261025
gpu_used: false
```

- [ ] **Step 3: Run a one-repeat pilot**

Add `--repeats 0` to the runner. Expected: three folds, finite importance, and exact validation index coverage. Compare runtime and selected modules before full execution.

- [ ] **Step 4: Run full E115**

Expected outputs:

- `q2_module_importance_folds_E115.csv.gz`;
- `q2_conditional_importance_folds_E115.csv.gz`;
- `q2_hierarchical_rankings_E115.csv`;
- `q2_top3_sufficiency_deletion_E115.csv`;
- `q2_anchor_recovery_E115.csv`;
- `q2_run_manifest_E115.json`.

The sufficiency/deletion file must contain outer-fold Top-3-only R²/MAE and delete-Top-3 R²/MAE. Top-3 selection for a validation fold must come only from that fold's training data.

- [ ] **Step 5: Commit**

```bash
git add '02_代码/Q2_关键化合物/src/hierarchical_importance.py' \
  '02_代码/Q2_关键化合物/src/run_e115_hierarchical_importance.py' \
  '02_代码/Q2_关键化合物/configs/E115_hierarchical_importance.yaml' \
  '02_代码/Q2_关键化合物/tests/test_hierarchical_importance.py' \
  '03_结果/Q2_关键化合物/' docs/experiment_log.md
git commit -m 'exp(q2): complete E115 hierarchical conditional attribution'
```

## Task 5: Evaluate E116 limited grouped multi-task sharing

**Files:**
- Modify: `02_代码/Q2_关键化合物/src/target_partial_effects.py`
- Modify: `02_代码/Q2_关键化合物/tests/test_target_partial_effects.py`
- Create: `02_代码/Q2_关键化合物/configs/E116_grouped_multitask.yaml`
- Create: `02_代码/Q2_关键化合物/src/run_e116_grouped_multitask.py`

- [ ] **Step 1: Add the target-group contract test**

```python
def test_target_groups_do_not_overlap_and_cover_registered_targets(self):
    groups = load_target_groups(CONFIG_PATH)
    flattened = [target for group in groups for target in group["targets"]]
    self.assertEqual(len(flattened), len(set(flattened)))
    self.assertTrue(set(flattened).issubset(set(load_target_codes())))
```

- [ ] **Step 2: Freeze only defensible groups**

The YAML contains explicit Aroma–Flavour pairs and the three broad families; no target may occur in more than one fitted group. Fit `MultiTaskElasticNet` within each group, and independent ElasticNet for ungrouped targets. Tune alpha and l1 ratio inside each outer-training fold.

- [ ] **Step 3: Run E116 and apply its gate**

Promote E116 into the conditional-effect family only if:

- macro R² decreases by no more than 0.003 from E105;
- macro MAE increases by no more than 0.002;
- bottom-quartile target R² improves;
- Top-3 bootstrap stability improves for the grouped targets;
- existing four direct anchors are not lost.

Write `q2_decision_E116.json` with every flag and exact path used.

- [ ] **Step 4: Commit**

```bash
git add '02_代码/Q2_关键化合物/src/target_partial_effects.py' \
  '02_代码/Q2_关键化合物/src/run_e116_grouped_multitask.py' \
  '02_代码/Q2_关键化合物/configs/E116_grouped_multitask.yaml' \
  '02_代码/Q2_关键化合物/tests/test_target_partial_effects.py' \
  '03_结果/Q2_关键化合物/' docs/experiment_log.md
git commit -m 'exp(q2): evaluate E116 limited multi-target sharing'
```

## Task 6: Implement E117 complementary stability and null controls

**Files:**
- Create: `02_代码/Q2_关键化合物/src/complementary_stability.py`
- Create: `02_代码/Q2_关键化合物/tests/test_complementary_stability.py`
- Create: `02_代码/Q2_关键化合物/configs/E117_stability_controls.yaml`
- Create: `02_代码/Q2_关键化合物/src/run_e117_stability_controls.py`

- [ ] **Step 1: Write failing complement and frequency tests**

```python
from complementary_stability import (
    stratified_complementary_pairs, selection_probability, empirical_fdp,
)

def test_complementary_halves_are_disjoint_and_cover_rows(self):
    style = np.repeat(["a", "b", "c"], [8, 6, 4])
    pairs = stratified_complementary_pairs(style, n_pairs=5, random_state=7)
    for left, right in pairs:
        self.assertEqual(set(left) & set(right), set())
        self.assertEqual(set(left) | set(right), set(range(len(style))))

def test_selection_probability_counts_both_halves(self):
    selections = [{0, 2}, {0}, {1}, {0, 1}]
    np.testing.assert_allclose(selection_probability(selections, 3), [0.75, 0.5, 0.25])

def test_empirical_fdp_is_bounded(self):
    self.assertEqual(empirical_fdp(real_selected=8, null_selected=2), 0.25)
```

- [ ] **Step 2: Implement balanced complementary splits**

Within each style, shuffle rows and split them as evenly as possible; alternate the extra row between halves across pairs. Concatenate style-specific halves so every pair is disjoint and covers all 250 rows. Save the exact row indices and seed.

Implement:

```python
def selection_probability(selection_sets, n_features):
    count = np.zeros(int(n_features), dtype=float)
    if not selection_sets:
        raise ValueError("at least one selection set is required")
    for selected in selection_sets:
        index = np.asarray(sorted(set(selected)), dtype=int)
        if len(index) and (index.min() < 0 or index.max() >= n_features):
            raise ValueError("selection index is out of bounds")
        count[index] += 1.0
    return count / len(selection_sets)

def complementary_pair_jaccard(left_sets, right_sets):
    if len(left_sets) != len(right_sets) or not left_sets:
        raise ValueError("left and right selections must have equal nonzero length")
    values = []
    for left, right in zip(left_sets, right_sets):
        union = set(left) | set(right)
        values.append(len(set(left) & set(right)) / len(union) if union else 1.0)
    return np.asarray(values, dtype=float)
def mb_pfer_bound(q, pi_threshold, p):
    return q*q / ((2*pi_threshold - 1)*p)
def empirical_fdp(real_selected, null_selected):
    return min(1.0, null_selected / max(real_selected, 1))
```

- [ ] **Step 3: Freeze E117 configuration**

```yaml
experiment: E117
pilot_pairs: 10
formal_pairs: 100
sampling: style_stratified_complementary_halves
selection_size_per_target: 3
stability_thresholds: [0.60, 0.70, 0.80, 0.90]
target_permutation_repeats: 50
shadow_feature_repeats: 20
target_permutation_scope: within_style
graph_and_ranking_recomputed_each_half: true
candidate_scope: strict_compounds_218
random_state: 20261025
gpu_used: false
```

- [ ] **Step 4: Run pilot, verify Monte Carlo convergence, then formal E117**

Formal outputs:

- `q2_cpss_selection_E117.csv`;
- `q2_cpss_pair_jaccard_E117.csv`;
- `q2_null_target_permutation_E117.csv.gz`;
- `q2_shadow_feature_audit_E117.csv`;
- `q2_error_control_E117.csv`;
- `q2_run_manifest_E117.json`.

The confirmed set uses the lowest threshold whose empirical FDP is at most 0.10. If no threshold qualifies, emit an empty confirmed set and retain all forced Top-3 entries as exploratory; never silently relax the threshold.

- [ ] **Step 5: Commit**

```bash
git add '02_代码/Q2_关键化合物/src/complementary_stability.py' \
  '02_代码/Q2_关键化合物/src/run_e117_stability_controls.py' \
  '02_代码/Q2_关键化合物/configs/E117_stability_controls.yaml' \
  '02_代码/Q2_关键化合物/tests/test_complementary_stability.py' \
  '03_结果/Q2_关键化合物/' docs/experiment_log.md
git commit -m 'exp(q2): complete E117 stability and null controls'
```

## Task 7: Build E118 three-family consensus and replacement gate

**Files:**
- Create: `02_代码/Q2_关键化合物/src/evidence_consensus.py`
- Create: `02_代码/Q2_关键化合物/tests/test_evidence_consensus.py`
- Create: `02_代码/Q2_关键化合物/configs/E118_evidence_consensus.yaml`
- Create: `02_代码/Q2_关键化合物/src/run_e118_evidence_consensus.py`

- [ ] **Step 1: Write failing fusion and gate tests**

```python
from evidence_consensus import fuse_evidence_families, replacement_decision

def test_family_fusion_prioritizes_cross_family_support(self):
    frame = pd.DataFrame({
        "feature_code": ["a", "b"],
        "predictive_rank": [1, 4],
        "partial_rank": [20, 2],
        "module_rank": [30, 3],
    })
    result = fuse_evidence_families(frame, rrf_k=60)
    self.assertEqual(result.iloc[0]["feature_code"], "b")

def test_replacement_requires_complete_output_and_two_stability_gains(self):
    flags = replacement_decision(
        complete_rows=150, prediction_ok=True,
        stability_gains={"frequency": True, "jaccard": True, "style": False},
        anchors_preserved=True, fdp_ok=True, leakage_ok=True,
    )
    self.assertTrue(flags["promoted"])
```

- [ ] **Step 2: Implement family fusion**

For each target and feature, compute family Top-3 and Top-10 support, then RRF:

```python
score = sum(1.0 / (rrf_k + rank) for rank in family_ranks)
```

Sort by Top-10 support, Top-3 support, RRF score, best rank and feature code. Apply these gates after ranking:

- CPSS confirmed/exploratory;
- bootstrap frequency and pairwise Jaccard;
- style-adjusted Top-10 and leave-style survival;
- positive median deletion loss;
- target predictability tier.

Do not add chemical grade or OAV to the score.

- [ ] **Step 3: Freeze replacement criteria**

```yaml
experiment: E118
families:
  predictive: E115_hierarchical_conditional_importance
  partial_effect: E114_or_promoted_E116
  graph_module: E113_if_promoted_else_E107
fusion: {method: support_aware_rrf, rrf_k: 60, weights: equal}
prediction_tolerance: {r2_drop_max: 0.003, mae_increase_max: 0.002, repeat_noninferior_min: 7}
required_stability_gains: 2
stability_metrics: [median_selection_frequency, pairwise_jaccard, leave_style_survival]
confirmed_fdp_max: 0.10
anchors_must_preserve_existing_four: true
chemical_evidence_in_ranking: false
candidate_scope: strict_compounds_218
```

- [ ] **Step 4: Run E118 and validate 50×3 output**

Write:

- `q2_evidence_family_rankings_E118.csv.gz`;
- `q2_final_top3_statistical_E118.csv`;
- `q2_replacement_comparison_E118.csv`;
- `q2_replacement_decision_E118.json`;
- `q2_run_manifest_E118.json`.

The test must assert exactly 50 targets, three ranks per target, 150 unique target-rank pairs, strict candidate eligibility, and complete compound-name mapping.

- [ ] **Step 5: Commit**

```bash
git add '02_代码/Q2_关键化合物/src/evidence_consensus.py' \
  '02_代码/Q2_关键化合物/src/run_e118_evidence_consensus.py' \
  '02_代码/Q2_关键化合物/configs/E118_evidence_consensus.yaml' \
  '02_代码/Q2_关键化合物/tests/test_evidence_consensus.py' \
  '03_结果/Q2_关键化合物/' docs/experiment_log.md
git commit -m 'exp(q2): resolve E118 evidence consensus gate'
```

## Task 8: Apply E119 chemistry audit without reranking

**Files:**
- Create: `02_代码/Q2_关键化合物/configs/E119_chemical_audit.yaml`
- Create: `02_代码/Q2_关键化合物/src/run_e119_chemical_audit.py`
- Modify: `02_代码/Q2_关键化合物/tests/test_chemical_evidence.py`

- [ ] **Step 1: Add the no-reranking test**

```python
def test_chemical_audit_preserves_statistical_target_rank_pairs(self):
    statistical = pd.DataFrame({
        "target_code": ["t", "t", "t"],
        "consensus_rank": [1, 2, 3],
        "feature_code": ["a", "b", "c"],
    })
    audited = append_chemical_evidence(statistical, CHEMISTRY_FIXTURE)
    pd.testing.assert_frame_equal(
        statistical[["target_code", "consensus_rank", "feature_code"]],
        audited[["target_code", "consensus_rank", "feature_code"]],
    )
```

- [ ] **Step 2: Reuse the E112 evidence rules on the frozen E118 table**

If E118 is not promoted, write a skipped E119 manifest and keep E112 as final. If promoted, calculate OAV only for calibrated raw values with same-unit thresholds, attach direct-anchor, source, aging, off-flavor and literature codes, then grade A/B/C/D without reordering rows.

- [ ] **Step 3: Run E119**

Expected promoted output: `q2_final_top3_E119.csv`, 150 rows, with all E118 statistical fields plus chemical evidence and limitations. Expected skipped output: `q2_run_manifest_E119.json` with `status=skipped` and the E118 rejection reasons.

- [ ] **Step 4: Commit**

```bash
git add '02_代码/Q2_关键化合物/configs/E119_chemical_audit.yaml' \
  '02_代码/Q2_关键化合物/src/run_e119_chemical_audit.py' \
  '02_代码/Q2_关键化合物/tests/test_chemical_evidence.py' \
  '03_结果/Q2_关键化合物/' docs/experiment_log.md
git commit -m 'exp(q2): complete E119 post-ranking chemistry audit'
```

## Task 9: Generate four traceable Q2 V2 figures

**Files:**
- Create: `02_代码/Q2_关键化合物/src/plot_q2_v2.R`
- Create: `02_代码/Q2_关键化合物/src/prepare_q2_v2_figure_data.py`
- Create: `02_代码/Q2_关键化合物/tests/test_q2_v2_artifacts.py`
- Create: `04_图表/Q2_关键化合物/v2/README.md`
- Create: `04_图表/Q2_关键化合物/v2/Q2_V2_图清单.csv`

- [ ] **Step 1: Write the artifact contract test**

The test asserts four PNG files, 300 dpi, landscape dimensions, non-empty source-data CSVs, no image-level title string in the R source, and an exact mapping between figure list and files.

- [ ] **Step 2: Prepare source data**

`prepare_q2_v2_figure_data.py` reads only promoted E113—E119 results, or E112 if the gate rejects the new route. It writes one CSV per panel and a JSON manifest containing input SHA256 hashes, row counts and selected result version.

- [ ] **Step 3: Implement the plots**

Generate:

1. `图1_偏相关图与稳定模块.png` — edge comparison plus co-clustering/module-level network;
2. `图2_预测性能与Top3有效性.png` — target-level paired deltas, Top-3 sufficiency and deletion loss;
3. `图3_稳定性与误选控制.png` — CPSS/style survival, new-old Jaccard and null separation;
4. `图4_强证据关系图谱.png` — preregistered 12—15 strongest relationships with direction, interval, grade and OAV.

Use `source("scripts/plot_style.R") + theme_mcup()`, 300 dpi, Chinese presentation text, bold panel labels, no gray facet strips, no internal overall title and legends outside the data region.

- [ ] **Step 4: Render, inspect and hash**

Run the R script twice and verify identical SHA256 hashes. Render each at 70% paper scale and visually inspect label overlap. Do not embed into DOCX until the user later requests paper assembly.

- [ ] **Step 5: Commit**

```bash
git add '02_代码/Q2_关键化合物/src/prepare_q2_v2_figure_data.py' \
  '02_代码/Q2_关键化合物/src/plot_q2_v2.R' \
  '02_代码/Q2_关键化合物/tests/test_q2_v2_artifacts.py' \
  '03_结果/Q2_关键化合物/figure_data_v2/' \
  '04_图表/Q2_关键化合物/v2/'
git commit -m 'fig(q2): produce evidence-first V2 figure checkpoint'
```

## Task 10: Audit, report and hand off

**Files:**
- Create: `docs/analysis/2026初赛/Q2_证据链优化报告.md`
- Modify: `docs/analysis/2026初赛/BZD与agent对照.md`
- Modify: `05_论文/Q2_关键化合物/03_方法与实验矩阵.md`
- Modify: `05_论文/Q2_关键化合物/04_结果与讨论.md`
- Modify: `05_论文/Q2_关键化合物/05_图表与表格清单.md`
- Modify: `docs/collab/board.md`
- Modify: `docs/collab/to_zcode.md`

- [ ] **Step 1: Run complete verification**

```bash
conda run -n mathorcup python -m unittest discover \
  -s '02_代码/Q2_关键化合物/tests' -p 'test_*.py' -v
Rscript '02_代码/Q2_关键化合物/src/plot_q2_v2.R'
git diff --check
```

Expected: all tests pass, four figures exist, E100—E112 files are unchanged, and the chosen result version is explicit.

- [ ] **Step 2: Write the decision report**

The report must contain:

- task and leakage boundary;
- E112 baseline and E113—E119 table;
- graph, prediction, partial-effect, stability, null-control, style and chemistry results;
- every replacement flag and exact failure reason;
- BZD/agent/agent2/agent3 comparison under explicit non-comparability notes;
- accepted route or retained E112 decision;
- unadopted methods and limitations.

Every number links to a machine-readable result path.

- [ ] **Step 3: Update paper handoff text**

Only update Markdown handoff materials. Do not produce or overwrite a Q2 DOCX in this task. Results must use the promoted version; if E118 rejects, keep E112 numbers and add the new experiments to the “未采纳方案” discussion.

- [ ] **Step 4: Commit and push**

```bash
git add '02_代码/Q2_关键化合物/' '03_结果/Q2_关键化合物/' \
  '04_图表/Q2_关键化合物/v2/' '05_论文/Q2_关键化合物/' \
  'docs/analysis/2026初赛/Q2_证据链优化报告.md' \
  'docs/analysis/2026初赛/BZD与agent对照.md' \
  docs/experiment_log.md docs/collab/board.md docs/collab/to_zcode.md
git commit -m 'collab(Q2-OPT): complete Q2 evidence optimization audit'
git push
```

- [ ] **Step 5: Verify handoff state**

```bash
git status --porcelain
git rev-parse HEAD
git rev-parse '@{upstream}'
```

Expected: empty status and identical local/upstream hashes.

## Plan self-review

- Spec coverage: Tasks 1—10 cover fold-local partial graphs, module stability, nonlinear and linear conditional effects, limited target sharing, CPSS, negative controls, three-family fusion, post-ranking chemistry, four figures and paper handoff.
- Scope control: E110 remains untriggered; chemistry never enters rank optimization; Q2 numerical execution is prohibited while Q1 E007 is running.
- Type consistency: graph matrices are 218×218; fold rankings are 30×50×218; final statistical and chemical tables are 50×3=150 rows; selection probabilities lie in [0,1].
- Leakage control: all preprocessing, graphs, modules, models, rank fusion inputs and conditional permutation models used for a validation fold are trained without that fold.
- Placeholder scan: no TBD/TODO/fill-later steps; every task has exact files, commands, expected outputs and commit boundaries.
