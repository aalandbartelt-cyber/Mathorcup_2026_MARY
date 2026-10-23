# 赛前准备说明（给 WSL Codex）

> 比赛 18:00 开始；赛题由用户直接粘贴。现在只做准备工作，**不要跑训练、不要占 GPU**。

## 1. 角色

- 你（WSL Codex）= **核心**：建模推理、模型选型、结果解读、代码/训练/评估、论文 LaTeX。
- ZCode（Windows）= 环境、数据、文档、复核、提交检查。
- dsh（Windows，DeepSeek Harness）= 论文评审与优化，只给建议，不改代码/结果。
- 边界见 `docs/collab/CODEX_职责说明.md`。

## 2. 工作区（全部在 `~/mathorcup`）

| 路径 | 内容 |
|---|---|
| `README.md` | 环境、协作、2026 提交总览 |
| `docs/M杯优秀论文特点、题型特点.md` | 章节结构、摘要模板、图表风格、技术栈、评估、40 篇逐篇卡片 |
| `docs/2026练习题精解与赛题技术映射.md` | 官方练习数据体检、解题方案、代码骨架 |
| `docs/官方参考学习资料精读.md` | 官方六大答卷问题、2025 A/B、2022 讲解 |
| `docs/比赛执行清单.md` | D-2/D-1、开赛 0–2h、提交 20 项 |
| `docs/2026提交规范速查.md` | 2026 命名（MCB2601605 → A2601605/B2601605） |
| `docs/自强平台操作指南.md` | A100 容器/GPU/同步/长任务/报错 |
| `docs/runbook.md` | 四天流程与坑清单 |
| `docs/templates/` | 分类/检测/分配结果模板、实验记录表 |
| `docs/collab/` | `board.md`、`to_codex.md`、`to_zcode.md`、`decisions.md`、`reviews/` |
| `06_交付/` | `00_官方模板`、`01_论文`、`02_结果`、`03_承诺书`、`04_支撑材料`、`提交清单.md` |
| `scripts/` | `train.sh`、`predict_to_csv.py`、`check_result.py`、`prepare_submission.py`、`eda_yolo.py`、`split_train_val.py`、`plot_style.R`、`plot_style.py`、`collab_msg.py`、`session_info.R` |
| skill 已装 | `~/.codex/skills/mary-modeling-contest-pipeline-mcup/` |

环境：conda `mathorcup`（torch 2.12 cu128、ultralytics 8.4、opencv）；R 环境 `r`（ggplot2/ragg）；GPU RTX 5070 Ti（12GB）。A100 用自强平台，见平台指南。

## 3. 18:00 前要做（只读）

1. 读 `docs/collab/CODEX_职责说明.md`、`docs/比赛执行清单.md`、`docs/2026提交规范速查.md`。
2. 确认 skill 可加载。
3. 环境自检：

```bash
conda activate mathorcup
python -c "import torch, ultralytics, cv2; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
Rscript -e 'source("scripts/plot_style.R"); cat("R theme OK\n")'
ls 06_交付 06_交付/00_官方模板与规范 docs/templates
python scripts/check_result.py --help | head -5
```

4. 检查 `06_交付/` 结构与官方模板是否齐全。
5. **不跑训练、不占 GPU**；`MSG-20261021-1209-96`（练习2 微调）保持 PAUSED。

## 4. 赛题来了之后，第一轮输出

1. 判断赛道 A/B 并给理由（数据形态/团队栈/算力）。
2. 逐问拆解：要求回答什么、输出什么、指标、约束、交付格式。
3. 数据清点命令（文件数/格式/类别/缺失/泄漏风险）。
4. 技术路线 2 条（主线 + 备选）与实验矩阵（基线/对比/消融）。
5. 时间表（7 天或压缩 4 天）与分工。
6. 需要 ZCode/用户配合的事项（数据上传、A100 开卡、权限）。
7. 明确“今天先做什么、今晚产出什么”。

## 5. 执行原则

- 先 baseline 再改进；每问先“任务本质 → 难点 → 对策”。
- 出图固定 R/ggplot2 `theme_mcup()`；Python 仅保底。
- 结果 CSV 用 `docs/templates/` + `scripts/check_result.py`；归档用 `scripts/prepare_submission.py`。
- 论文：首页摘要 + 第二页目录 + 正文 ≤30 页 + 附录全代码；摘要每问带数字。
- 静默协作：只在交接/阻塞/决策时写 `docs/collab/to_zcode.md`；日常进度写 `board.md`。

## 6. 现在回复

准备完成 / 环境三行 / skill 是否可加载 / 还缺什么。然后待命。
