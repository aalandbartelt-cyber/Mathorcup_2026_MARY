# Q1 NNLS Preparation Implementation Plan

> **For agentic workers:** use executing-plans inline. User already authorized autonomous implementation; experimental entry conditions remain binding.

**Goal:** Produce traceable OOF complementarity evidence and a tested NNLS stacking implementation while E007 continues unchanged.

**Architecture:** Separate read-only diagnostics, pure NNLS fitting, and a gated outer-fold runner. Reuse E005 base-model functions without editing them. Write new artifacts only in q1-nnls-opt.

**Tech Stack:** NumPy, SciPy NNLS, pandas, scikit-learn, unittest; CPU diagnostics/synthetic tests.

## Task 1: Read-only OOF diagnostics

Files: `02_代码/Q1_感官评分/src/stacking_diagnostics.py`, `tests/test_stacking_diagnostics.py`.

- [x] RED: identical predictions have no exclusive rescue; complementary errors give positive rescue in both directions; shuffled beer/target ordering raises ValueError.
- [x] Implement `validate_oof_alignment(reference, candidate)` and `pairwise_complementarity(y_true, predictions, target_codes)`; return per-repeat/per-target records, never fit or choose weights.
- [x] Verify positive affine predictions keep `ordered_proxy_metrics` unchanged.
- [x] Run the diagnostic entry point on existing E003/E005 archives, exporting `q1_complementarity_*` and input hashes, explicitly `diagnostic_only=true`.

Test command:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/mary/miniconda3/envs/mathorcup/bin/python -m unittest '02_代码/Q1_感官评分/tests/test_stacking_diagnostics.py' -v
```

## Task 2: NNLS primitives

Files: `src/nnls_stacking.py`, `tests/test_nnls_stacking.py`, `configs/E009_nnls_stacking.yaml`.

- [x] RED: `fit_nnls_stacking(stack, y, groups, scope, shrinkage)` recovers a negative free intercept with nonnegative model weights.
- [x] Implement centered/scaled training-only shared/group/target fitting using `scipy.optimize.nnls`; fixed 0.5 shrinkage for group/target scopes.
- [x] Test validation predictions cannot change fitted weights, constant predictions safely fall back to the training mean, and malformed tensors/groups fail explicitly.
- [x] Register E009 scopes, primary group candidate, E005 source config, 10×3/inner-3, fixed random seed and five unchanged thresholds.

Test command:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/mary/miniconda3/envs/mathorcup/bin/python -m unittest '02_代码/Q1_感官评分/tests/test_nnls_stacking.py' -v
```

## Task 3: Entry guard and integration preparation

Files: `src/run_e009_nnls_stacking.py`, `tests/test_e009_runner.py`.

- [x] RED: missing E007 complete manifest rejects actual-data execution before any outputs or model fitting.
- [x] Reuse `_candidates`, `_fit_base` and `_parameter_key` from E005; generate inner OOF from each outer-training fold, fit all three pre-registered NNLS scopes, predict unseen outer validation once.
- [x] Cache per-fold predictions and parameter rows with exact input/config/source/CV hashes; reject incomplete or mismatched caches.
- [x] Test the fold helper only on synthetic arrays; import and `--help` must not fit anything.
- [x] Run Q1 regression tests and document evidence. Actual E009 execution remains gated on completed E007 audit and diagnostic support.

Verification checkpoint: 83 Q1 tests passed, including 16 new diagnostics/NNLS/runner
tests and a temporary-directory synthetic screening+resume integration test. Existing
OOF diagnosis wrote seven CSV files and one manifest; no E009 actual-data training ran.
