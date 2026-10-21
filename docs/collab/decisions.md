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

## D-003 双 AI 分工
- 决定：Codex（WSL）负责核心开发与实验；ZCode（Windows）负责环境运维、数据整理、文档、复核与替补。
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
