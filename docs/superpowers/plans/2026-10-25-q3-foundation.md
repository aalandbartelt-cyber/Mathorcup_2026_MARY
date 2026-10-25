# Q3 Foundation Implementation Plan

> **For agentic workers:** Use executing-plans inline; user authorized autonomous preparation. No subagents, no E009 monitoring, no model training.

**Goal:** Build a reproducible Q3 input contract and pure audit/evaluation kernels before any generator experiment.

**Architecture:** Three independent modules with explicit arrays/tables, a no-training E200 writer, and synthetic boundary tests. Existing Q1/Q2 code and outputs are read-only.

**Tech Stack:** Existing mathorcup Python, NumPy, pandas, SciPy, unittest; no new dependencies.

## Task 1: Data and units

Files: `02_代码/Q3_风格生成/src/q3_contract.py`, `tests/test_q3_contract.py`, `configs/E200_contract.yaml`.

- [x] RED: shuffled beer alignment succeeds; duplicate/style mismatch fails; synthetic raw/log10/unknown fields are distinguished, individual mismatches/zeros/missing remain unknown; unit mismatch fails.
- [x] Implement `align_by_beer(reference, other, name)`, `feature_contract(chemistry, calibrated, metadata, minimum_pairs=3, minimum_match=.95)`, `defect_registry(metadata)`, `source_defect_table(chemistry, calibrated, fields, source_style)`.
- [x] Required example: a raw column with one conflicting positive pair keeps `scale_in_1_csv=raw` when match≥.95, but the cell is `reference_conflict`; calibrated zero never overwrites source NA.
- [x] GREEN command:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/mary/miniconda3/envs/mathorcup/bin/python -m unittest discover -s 02_代码/Q3_风格生成/tests -p test_q3_contract.py -v
```

## Task 2: Explicit defect scenarios

Files: `src/q3_constraints.py`, `tests/test_q3_constraints.py`.

- [x] RED: `source=1, candidate=1.5, threshold=2` fails C1, passes C2, fails C3; `source=3,candidate=2.5,threshold=2` passes C1/C2 and fails C3. Missing sources/thresholds return unknown, negative/NaN candidates invalid.
- [x] Implement `audit_concentrations` with limits source/max/min, nullable passed flags and candidate-minus-limit positive part; fixed numerical tolerance only.
- [x] Implement `project_upper_bounds` as minimum where bounds known, preserving lower values and unknown bounds; implement migration summary with explicit no_target_gain/unknown/proxy rejection.
- [x] GREEN: run `test_q3_constraints.py`; verify input arrays not mutated and unknowns excluded from known-only rate while overall status remains incomplete.

## Task 3: Independent distribution diagnostics

Files: `src/q3_evaluation.py`, `tests/test_q3_evaluation.py`.

- [x] RED: identical finite samples yield zero W1/energy; translation positive; unbalanced sample counts, invalid scales and invalid k fail; constants report no correlation pairs; training copies have duplicate fraction1.
- [x] Implement `distribution_diagnostics(real, generated, training, scale, knn_k=1)` using caller-provided positive scales, per-column Wasserstein and normalized pairwise distances.
- [x] Energy uses `2*mean(d(real,gen))-mean(d(real,real))-mean(d(gen,gen))`; name it V-statistic, not scipy's 1D energy distance. Coverage uses real kth-neighbor radii, with self distances excluded.
- [x] GREEN: run `test_q3_evaluation.py`; no sklearn classifier, generator or data-dependent tuning inside diagnostics.

## Task 4: E200 writer and handoff

Files: `src/run_e200_contract.py`, `tests/test_e200_runner.py`, Q3 README, `05_论文/Q3_风格生成/03_E200数据核验与口径补充.md`.

- [x] RED: writer creates only declared contract files in a temporary directory; duplicate/missing CV slots fail; manifest declares `model_fitted=false` and contains source/data hashes; no full-data imputation output.
- [x] Implement `run_contract(root, output_dir)` reading attachments1/4/8, Q1 frozen CV and compound-name dictionary; call pure helpers, validate actual dimensions, write descriptive CSV and manifest via `--write`.
- [x] Run E200 once on actual attachments. Verify input hashes unchanged and exports reflect37 log10/33 raw/161 unverified fields plus conflicts/unknowns; do not claim all70 fields have every cell certified.
- [ ] Run all Q3 tests and Q1 27 baseline tests; `git diff --check`; commit/push branch and record path/limitations in main task board.

## Self-review

The plan covers only E200 and reusable pure primitives. No generator fit, prediction result, C2ST, graph fitting, GPU use, new dependency or scientific performance claim is authorized by this preparation checkpoint. Complete each module's red/green evidence before using it on the actual contract.

Verification checkpoint: 31 Q3 tests passed, including regressions for actual negative reference values, shuffled DataFrame indices, unsafe vector broadcasting, string booleans, nonfinite tolerances and normalization overflow. E200 exported exactly8 files with unchanged raw input hashes; its83 exceptions are80 positive-pair discrepancies plus3 negative reference values. No Q3 model was trained.
