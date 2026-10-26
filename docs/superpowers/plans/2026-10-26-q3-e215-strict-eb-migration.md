# Q3 E215 Strict EB Migration Implementation Plan

**Goal:** Add a separately audited empirical-Bayes migration endpoint while preserving all frozen Q3 results.

**Architecture:** A small mathematical kernel computes shrunken style geometry and a deterministic active-set Mahalanobis projection. A read-only runner builds the 33 source–target endpoints and seven fixed paths per pair; the first checkpoint reports chemical and geometric feasibility only. Sensory selection remains a separate later gate.

**Tech stack:** Python 3.10, NumPy, pandas, scikit-learn LedoitWolf, pytest, YAML.

---

### Task 1: Register the fixed protocol

**Files:** `02_代码/Q3_风格生成/configs/E215_strict_eb_migration.yaml`, `docs/superpowers/specs/2026-10-26-q3-e215-strict-eb-migration-design.md`.

- [ ] Fix the source/target styles, `κ_μ=15`, `κ_Σ=30`, C1, seven path values, and Q1/no-GPU flags before numerical output.
- [ ] Add a config test that rejects changed styles, shrinkage, λ and interface flags.
- [ ] Commit and push the registered configuration.

### Task 2: Test and implement mathematical kernel

**Files:** `02_代码/Q3_风格生成/tests/test_eb_migration.py`, `02_代码/Q3_风格生成/src/eb_migration.py`.

- [ ] Write tests for source identity at λ=0, positive-definite covariance, exact KKT projection, non-active coordinates following covariance, and known upper bounds.
- [ ] Run the test and verify it fails because the kernel is absent.
- [ ] Implement fold-independent post-selection style geometry and four-dimensional active-set projection.
- [ ] Run the test and the complete Q3 test suite.
- [ ] Commit and push the kernel.

### Task 3: Build 33-pair geometric checkpoint

**Files:** `02_代码/Q3_风格生成/src/run_e215_strict_eb_migration.py`, `02_代码/Q3_风格生成/tests/test_e215_runner.py`, `03_结果/Q3_风格生成/q3_*_E215.*`.

- [ ] Test exact 11×3 coverage, input/Q1 identity, C1 known/unknown separation and no alteration of source λ=0.
- [ ] Run once against frozen E200/E204 inputs and write endpoints, paths, diagnostics and SHA-256 manifest.
- [ ] Compare target gain and support with E204 without using Q1/ElasticNet to select a winner.
- [ ] Audit and commit/push the checkpoint.

### Task 4: Decide whether a locked sensory search is warranted

- [ ] Report the complete 33-pair denominator and any out-of-support or direct-constraint failures.
- [ ] If geometric feasibility is adequate, register a separate candidate-selection/lock specification before reading independent model-family audit scores.
- [ ] Preserve existing E205B/E209 candidate counts and paper language until a new locked audit is complete.
