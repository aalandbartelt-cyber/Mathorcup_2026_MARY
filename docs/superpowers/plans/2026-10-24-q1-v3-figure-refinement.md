# Q1 V3 Figure Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a traceable, visually refined Q1 V3 figure set without changing E001—E003 models, metrics, selected representative targets, or frozen V1/V2 artifacts.

**Architecture:** Keep the frozen CSV-to-R plotting pipeline. Extend the existing R script with evidence-focused encodings, including a non-gray model palette and a three-panel R²/Spearman target atlas, render to an isolated `final_v3_draft/` directory, and validate dimensions, manifests, deterministic hashes, frozen-version integrity, and placed-size/grayscale previews with a dedicated Python audit.

**Tech Stack:** R 4.x, ggplot2, dplyr, tidyr, readr, patchwork, ragg/theme_mcup; Python 3.10, Pillow, pandas, hashlib; Git.

---

## File map

- Modify `02_代码/Q1_感官评分/src/plot_q1_drafts.R`: implement V3 encodings while preserving configurable output directories.
- Create `02_代码/Q1_感官评分/tests/test_q1_v3_artifacts.py`: audit rendered files, source counts, dimensions, DPI, manifests, and frozen-version hashes.
- Modify `.research/figures/Q1_figure_plan.md`: record V3 visual arguments and review risks.
- Modify `.research/figures/Q1_source_data_manifest.json`: route outputs to `final_v3_draft/`.
- Create `04_图表/Q1_感官评分/final_v3_draft/`: five PNGs, figure list, and SHA256 manifest.
- Modify `04_图表/Q1_感官评分/README.md`, `task_plan.md`, `findings.md`, `progress.md`: record status and decisions.
- Modify `docs/collab/to_zcode.md` and `docs/collab/board.md`: send the V3 review checkpoint.

### Task 1: Add the failing artifact audit

**Files:**
- Create: `02_代码/Q1_感官评分/tests/test_q1_v3_artifacts.py`

- [x] **Step 1: Write the audit test**

The test defines the expected five PNG names and dimensions, verifies approximately 300 dpi, requires a five-row figure list and valid SHA256 manifest, asserts source counts of 50 targets/22 styles/the fixed representative targets, and verifies both frozen checksum manifests.

```python
EXPECTED = {
    "图1_模型对比.png": (2460, 1305),
    "图2_目标分布与逐目标R2.png": (3360, 1890),
    "图3_OOF观测预测.png": (2640, 1260),
    "图4_风格误差与量表映射.png": (3150, 1770),
    "附图S1_残差诊断.png": (2640, 1260),
}
```

- [x] **Step 2: Run the audit and verify it fails before rendering**

Run:

```bash
conda run -n mathorcup python -m unittest 02_代码/Q1_感官评分/tests/test_q1_v3_artifacts.py -v
```

Expected: FAIL because `04_图表/Q1_感官评分/final_v3_draft/` does not exist.

### Task 2: Implement the V3 visual encodings

**Files:**
- Modify: `02_代码/Q1_感官评分/src/plot_q1_drafts.R`

- [x] **Step 1: Add the Figure 1 mean markers**

Add after jitter points in both model-comparison panels:

```r
stat_summary(fun = mean, geom = "point", shape = 23, size = 2.35,
             stroke = 0.45, fill = "white", colour = "black")
```

- [x] **Step 2: Replace Figure 2 with a three-panel target-performance atlas**

Map the original five target groups into three layout groups:

```r
target_metrics <- target_metrics %>%
  mutate(panel_group = case_when(
    target_group == "Aroma" ~ "香气",
    target_group == "Flavour-core" ~ "风味",
    TRUE ~ "味觉·口感·总体"
  ))
```

Create one horizontal dumbbell panel per layout group. Each target row connects R² (deep-blue filled circle) and Spearman (orange open diamond) with a light neutral segment. Sort targets by R² within each panel, use the same 0—0.9 x-axis, and combine the 15/15/20-row panels with equal widths. Keep all 50 targets.

- [x] **Step 3: Add OLS trends and Chinese panel labels to Figure 3**

Change the three labels to `高可预测：酸度`, `中可预测：4VG`, and `低可预测：陈旧酒花味`. Add `geom_smooth(method="lm", formula=y~x, se=TRUE)` below the points, using the existing tier colors and a low-alpha confidence ribbon. Keep the dashed 1:1 line and fixed shared scales.

- [x] **Step 4: Add Figure 4 benchmark and direct curve labels**

Compute `mean_oof_mae <- mean(target_metrics$mae)` from the 50 target-level mean-OOF metrics, then add a vertical line at that value and plain-text annotation `总体 MAE=0.339`. Do not reuse Figure 1's 10-repeat mean MAE=0.3422. Map four line types to the scale curves. Build a four-row direct-label frame by selecting curve points nearest grades 5.70, 5.35, 5.00, and 4.65, draw colored text at those points, and suppress separate color/linetype legends.

- [x] **Step 5: Keep Supplement S1 geometry unchanged**

Reuse the Chinese facet labels from Figure 3; do not change residual points, zero line, or LOESS smoother.

### Task 3: Render and inspect V3

**Files:**
- Create: `04_图表/Q1_感官评分/final_v3_draft/*.png`
- Create: `04_图表/Q1_感官评分/final_v3_draft/Q1_图清单.csv`

- [x] **Step 1: Render with the isolated output directory**

```bash
Q1_FIGURE_DIR=04_图表/Q1_感官评分/final_v3_draft conda run -n r Rscript 02_代码/Q1_感官评分/src/plot_q1_drafts.R
```

Expected: `Generated 4 main figures and 1 supplementary figure`.

- [x] **Step 2: Inspect all figures at original resolution**

Reject and revise if text crosses a border, direct labels overlap, group colors are ambiguous, or Figure 2 target names are unreadable.

- [x] **Step 3: Generate 70% and grayscale contact sheets for QA**

Use ImageMagick only for temporary previews under `/tmp`; do not commit these previews. Confirm V3 remains readable and curves remain distinguishable without color.

- [x] **Step 4: Generate the SHA256 manifest**

Write all five PNG hashes and `Q1_图清单.csv` to `final_v3_draft/SHA256SUMS.txt`, then verify with `sha256sum -c`.

### Task 4: Validate determinism and frozen-version integrity

**Files:**
- Test: `02_代码/Q1_感官评分/tests/test_q1_v3_artifacts.py`

- [x] **Step 1: Run the artifact audit**

```bash
conda run -n mathorcup python -m unittest 02_代码/Q1_感官评分/tests/test_q1_v3_artifacts.py -v
```

Expected: PASS for artifact, source-count, dimension, DPI, and frozen-hash checks.

- [x] **Step 2: Verify deterministic redraw**

Record V3 hashes, rerun the R command, and check that every hash is unchanged.

- [x] **Step 3: Run all Q1 regression tests and whitespace checks**

```bash
conda run -n mathorcup python -m unittest discover -s 02_代码/Q1_感官评分/tests -p 'test_*.py' -v
git diff --check
```

Expected: all Q1 tests pass and `git diff --check` prints nothing.

### Task 5: Update the figure contract and review checkpoint

**Files:**
- Modify: `.research/figures/Q1_figure_plan.md`
- Modify: `.research/figures/Q1_source_data_manifest.json`
- Modify: `04_图表/Q1_感官评分/README.md`
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`
- Modify: `docs/collab/to_zcode.md`
- Modify: `docs/collab/board.md`

- [x] **Step 1: Update provenance documents**

Record the quantile-interval encoding, OLS calibration trend, overall-MAE reference, direct labels, V3 output directory, and V1/V2 freeze.

- [x] **Step 2: Send the V3 figure list**

Append one `handoff` message listing all five V3 paths, the visual change in each figure, unchanged model numbers, and the condition that Word generation remains paused pending user approval.

- [x] **Step 3: Commit and push the V3 checkpoint**

```bash
git add 02_代码/Q1_感官评分 .research/figures 04_图表/Q1_感官评分 docs/collab task_plan.md findings.md progress.md
git commit -m "collab(MSG-ID): refine Q1 V3 figures"
git push
```

Expected: clean `main` synchronized with `origin/main`; report resulting HEAD and review paths.
