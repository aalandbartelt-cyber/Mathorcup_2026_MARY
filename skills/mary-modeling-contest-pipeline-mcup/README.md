# mary-modeling-contest-pipeline-mcup

MathorCup 大数据竞赛（MCup）特化版流水线 skill。

协作角色：**Codex**（WSL，核心：建模推理/选型/结果解读/实现）、**ZCode**（Windows，环境/复核）、
**dsh（DeepSeek Harness，Windows，论文评审与优化）**。dsh 不承担建模与选型；评审意见经 Codex/ZCode 落实。

- 继承：`mary-modeling-contest-pipeline` 的通用方法（口径统一、逐问求解、验证优先、R 出图、交接包、提交红线）。
- 替换：赛制层为 MathorCup 大数据竞赛——A/B 双赛道、初赛/复赛、结果 CSV、30 页论文、官方六大答卷问题。
- 依据：40 篇 2021–2025 优秀论文（1567 页）+ 9 份官方赛题/规范 + 2025/2026 真题。

## 安装位置

- WSL / Codex：`~/.codex/skills/mary-modeling-contest-pipeline-mcup/`
- Windows / Codex：`C:\Users\<user>\.agents\skills\mary-modeling-contest-pipeline-mcup\`
- 版本库（canonical）：`<MathorCup 仓库>/skills/mary-modeling-contest-pipeline-mcup/`

## 结构

```
SKILL.md                     主文件（何时用、三铁律、五阶段、题型速查、Top10 错误）
references/
  01_赛制与赛道选择.md        A/B 差异、决策树、赛事参数、格式红线、时间轴
  02_工程与数据契约.md        目录、命名、数据对账、划分、结果模板、Git
  03_题型解法库.md            CV/表格/时序/优化/无监督 的基线→改进→指标→写法
  04_实验与评估.md            六维评估、指标选择、消融/对比、误差分析、数字一致性
  05_出图规范_R.md            theme_mcup、四类图、字号画布、QA、反模式
  06_论文写作与摘要.md        30 页结构、摘要四段式、每问模板、参考文献
  07_提交与红线.md            结果格式、审计脚本、20 项清单、元数据清理
  08_协作与恢复.md            Codex/ZCode 信箱、静默模式、阻塞、恢复
  09_案例库与官方口径.md      2025 A/B、2022 讲解、2026 练习、40 篇统计、官方六问
  10_论文评审与优化.md        dsh 专用：六维 rubric、P0/P1/P2、评审输出模板
  11_自强平台操作指南.md      A100 容器/GPU 分配/代码同步/长任务/报错速查
  12_AI工具合规.md            2026 AI 规定：声明/参考文献/标注/支撑材料 PDF
assets/
  project_tree.txt            目录骨架
  paper_outline.md            论文骨架
  checklists.md               六大问题+20 项+图表 QA
  r_theme_mcup.R              出图主题（复制自项目 scripts/plot_style.R）
  result_templates/           分类/检测/分配/分批/实验记录
scripts/
  new_mcup_project.py         一键建骨架
  audit_mcup_submission.py    提交前审计（页数/元数据/结果格式/覆盖）
  collab_msg.py               双 AI 消息工具
```

## 快速开始

```bash
python scripts/new_mcup_project.py --root ~/mcup2026 --track A
# 填 00_赛题与规则/赛事参数.md 与 选题结论.md
# 数据放 01_数据/，完成 数据对账表.md
# 出图：source assets/r_theme_mcup.R
# 提交前：
python scripts/audit_mcup_submission.py --root ~/mcup2026 \
  --paper 06_交付/论文.pdf --result 03_结果/result.csv --images-dir 01_数据/test
```

## 与基础 skill 的差异

| 维度 | 基础版（国赛） | 本版（MathorCup） |
|---|---|---|
| 赛道 | 单赛题 A/B/C | **A 视觉 / B 表格，选错取消资格** |
| 时长 | 约 4 天 | 官方 7 天节奏（可压 4 天） |
| 首页 | 摘要专用页 | 题目+摘要+关键词；**第二页目录** |
| 交付 | 论文+支撑包 | 论文+**结果 CSV**+承诺书 |
| 出图 | R（雅黑+Times，400dpi） | **R（宋体+Times，300dpi）**；Python 保底 |
| 评估 | 评分线工具 | 六维评估 + 官方六大答卷问题 |
| 协作 | 交接包 | 文件信箱（docs/collab，静默模式） |

## 维护

- 赛制变化：只改 `references/01` 与 `07`，其余方法层不动。
- 主题/模板：从比赛仓库 `scripts/plot_style.R`、`docs/templates/` 同步到 `assets/`。
- 新增案例：追加到 `references/09`，不要改 SKILL.md 的流程。
