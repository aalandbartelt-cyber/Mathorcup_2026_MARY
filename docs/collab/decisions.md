# 已定事项（Decisions）

> 只追加，不删除；编号 D-xxx。每条写明“决定 / 理由 / 影响文件”。

## D-001 出图栈：R/ggplot2 首选，Python 保底
- 时间：2026-10-21
- 决定：比赛出图固定用 R（`scripts/plot_style.R` 的 `theme_mcup()` 与四类图表函数）；matplotlib 版（`scripts/plot_style.py`）仅作保底。
- 理由：双栈并排对比（`figures/demo_style_compare.png`）后，团队认为 R 版观感更好。
- 影响：docs/M杯优秀论文特点、题型特点.md §4.4/§4.5.6；README；docs/runbook.md。

## D-002 赛程与开赛时间
- 时间：2026-10-21
- 决定：按官方讲座口径，预计 **2026-10-23 18:00** 开赛（saikr.com/vse/bigdata2026）；赛前按 D-2/D-1 清单执行。
- 影响：docs/比赛执行清单.md。

## D-003 三方分工
- 决定：Codex（WSL）负责核心工作（建模推理、模型选型、结果解读、实现与论文）；
  ZCode（Windows）负责环境运维、数据整理、文档、复核与替补；
  **dsh（DeepSeek Harness，Windows）负责论文评审与优化**，能力与 ZCode 相近但分工不同。
- 边界：dsh 不承担建模推理、选型、结果解读；其评审意见以 `review` 消息进入协作通道，由 Codex/ZCode 落实。
- 沟通方式：本目录文件信箱（见 README.md），不依赖对话上下文。

## D-004 代码与资产同步
- 决定：代码/文档/模板走 git（GitHub `aalandbartelt-cyber/Mathorcup_2026_MARY`）；数据、权重、参考材料不入库，本地保存。
- 影响：.gitignore；docs/env_r.txt / env_py.txt 入库用于复现。

## D-005 结果文件与校验
- 决定：使用 `docs/templates/` 三套模板；提交前必须跑 `scripts/check_result.py`。
- 影响：docs/比赛执行清单.md §6。

## D-006 AI 协作协议
- 决定：采用 `docs/collab/` 文件信箱；单一写入者原则；状态机 OPEN→ACK→DONE/BLOCKED。
- 影响：docs/collab/README.md、board.md、to_codex.md、to_zcode.md。

## D-007 2026 提交规范与队号
- 决定：按 2026-10 修订版规范执行；队号 **MCB2601605**，文件名用 **A2601605 / B2601605**（题号+7位数字）。
- 交付：论文 `A2601605.pdf`、承诺书 `A2601605承诺书`、支撑材料 `A2601605附件.rar|zip`；提交入口 `https://www.saikr.com/vse/bigdata2026`。
- 目录：`06_交付/`（00 官方模板、01 论文、02 结果、03 承诺书、04 支撑材料）；清单 `06_交付/提交清单.md`；速查 `docs/2026提交规范速查.md`。

## D-008 AI 工具合规（2026-08 规定）
- 决定：按 2026 年 8 月《人工智能工具使用规定（试行）》执行：论文参考文献后声明、参考文献列 AI 工具、正文/图表/代码标注、支撑材料提供 `AI工具使用详情说明.pdf`。
- 合规红线：核心建模、算法设计、求解与分析由队员独立完成；AI 作为辅助并留痕。
- 影响：`docs/AI工具使用规定_2026.md`、`06_交付/05_AI工具使用说明/`、`docs/templates/paper_ai_declaration.md`、skill `references/12`。

## D-009 2026 初赛锁定赛道 A
- 时间：2026-10-23
- 决定：锁定赛道 A（啤酒化合物分析与风味预测）；首选 PLS2/MultiTaskElasticNet、bootstrap 稳定归因和条件高斯 copula 的小样本稳健主线，树模型与条件 VAE 仅在交叉验证达到门槛后作为增强。
- 理由：A 已形成 250×231 化学特征、250×50 感官目标、相关矩阵和阈值的完整闭环；B 缺少事件级触发标签，且外部时空数据、GIS 与仿真依赖明显更重。
- 待澄清：`4.csv` 恢复 0—6 量表的规则；`2.csv` 的 Pearson/Spearman 口径；是否另有结果文件模板。
- 影响：后续实验、论文文件名与支撑材料均按 `A2601605` 组织；详见 `docs/analysis/2026初赛/首轮赛道判断.md`。
