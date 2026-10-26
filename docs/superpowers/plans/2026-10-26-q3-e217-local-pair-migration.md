# Q3 E217 Local Pair Migration Implementation Plan

> **For agentic workers:** Execute inline in the existing isolated `q3-foundation` worktree. Do not dispatch subagents or modify another worktree.

**Goal:** Test one fixed local two-anchor migration proposal while preserving all existing Q3 decisions and the 33-pair denominator.

**Architecture:** A small pure module chooses deterministic local anchor pairs. A runner reads the frozen E204 pool and Q1 v5 predictor caches, writes resumable per-source checkpoints, then locks one primary candidate per source-target group. A separate audit may use ElasticNet only after the lock is committed.

**Tech Stack:** Python 3.10, numpy/pandas/scipy/sklearn, pytest, Git; existing frozen Q1 adapter and Q3 constraints.

---

### Task 1: Freeze protocol and pure candidate construction

**Files:** Create `02_代码/Q3_风格生成/configs/E217_local_pair_migration.yaml`, `src/local_pair_migration.py`, `tests/test_local_pair_migration.py`.

- [ ] Write tests first for 24 nearest IDs with deterministic tie breaks; exactly 276 pair midpoints; 7 λ per midpoint; source λ=0 identity; no mutation of the E204 pool. A four-point fixture must verify pair order and exact midpoints.
- [ ] Run the focused test and observe the missing-function failure.
- [ ] Implement `nearest_pairs(pool, source, scale, sample_ids, k=24)` returning `(pair_ids, midpoint_values)` and `fixed_paths(source, midpoints, lambdas)` returning the immutable 7-point paths. Keep scale/missing/shape checks explicit.
- [ ] Run focused tests, then commit pure code, config, spec and plan before numerical search.

### Task 2: Frozen NNLS search and candidate lock

**Files:** Create `src/run_e217_local_pair_search.py`, `tests/test_e217_search.py`; write `03_结果/Q3_风格生成/q3_*_E217.*` only.

- [ ] Test that a group with unsupported paths cannot be selected, that zero feasible candidates is preserved, that minimum RMS change wins deterministic ties, and that a 33-group result rejects duplicate/missing identities. Observe each test fail before code.
- [ ] Implement a source-level checkpoint runner patterned on E205B but with no auditor inputs. Validate E204 pool SHA/shape and Q1 v5 registry/cache identity. Build 11×3×276×7=63,756 paths using the fixed generator; verify raw direct nonnegativity and C1 known upper bounds, preserve unknowns.
- [ ] Compute source-fold training support first. Predict Stale hops only for numerically valid, gain≥1%, support≥0.9 paths with the frozen NNLS cache; verify single-target prediction against the frozen OOF within 1e-10. Save all paths with rejection reason, 33 selections, locked spectra and SHA-bound per-source checkpoints.
- [ ] Run one-source checkpoint to inspect only runtime/integrity; do not change the frozen config. Continue to 11/11. Run focused/full Q3 tests, audit hashes, commit/push the lock. No ElasticNet access during this task.

### Task 3: Once-only post-lock audit and decision

**Files:** Create `src/audit_e217_local_pair_search.py`, `tests/test_e217_audit.py`, `docs/analysis/2026初赛/Q3_E217局部双锚点迁移正式审计报告.md`.

- [ ] Test that the auditor rejects an uncommitted or changed lock and never promotes unknown direct measurements. Observe the intended failure before implementation.
- [ ] Load committed lock bytes from Git and verify all input/output SHA-256. Recompute primary-only selection identities. If no locked spectra, skip ElasticNet and write the zero-candidate audit; otherwise reconstruct the frozen ElasticNet with E205B's established fold parameters and compare source/candidate predictions across all 10 repeats, retaining the entire 33-group denominator.
- [ ] Compare by target with E205B main 2/33 and E209 Lambic main 0/11 without changing old outputs. Apply the spec's beer=230→Lambic exploratory criterion. Write direct C1/C2/C3 unknown status, no physical-safety claim.
- [ ] Run full Q3 tests, direct hash/identity audit, `git diff --check`, inspect worktree status; then commit/push code, result and report. Update `docs/collab/board.md` with the decision.
