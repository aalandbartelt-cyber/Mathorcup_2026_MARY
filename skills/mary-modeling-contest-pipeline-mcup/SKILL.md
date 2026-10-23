---
name: mary-modeling-contest-pipeline-mcup
description: Use when running the MathorCup 大数据竞赛 (MathorCup Big Data Challenge, 赛道 A 视觉 / 赛道 B 表格) or a similar two-track data-modeling contest — covering 赛道选择、数据对账、CV/表格/时序/优化建模、R 出图、30 页论文组装、结果文件生成与校验、官方红线检查、双 AI 文件协作。Also use when a MathorCup submission must be audited (赛道与内容不符 / 结果 CSV 格式错 / 论文超页 / 附录代码跑不通 / 元数据泄露身份).
---

# MARY 数模竞赛流水线 · MathorCup 大数据竞赛特化版（MCup）

本 skill 是 `mary-modeling-contest-pipeline`（2026 国赛 C 题实战版）的 **MathorCup 大数据竞赛特化版**。
方法层继承：口径统一、逐问求解、验证优先、R 出图、交接包、提交红线；
赛制层替换为：**A/B 双赛道、初赛/复赛、结果 CSV、30 页论文、官方六大答卷问题**。
数据与案例来自 40 篇 2021–2025 优秀论文（1567 页）+ 9 份官方赛题/规范 + 2025/2026 真题。

## 何时用

- 拿到 MathorCup 大数据赛题，要在 A（视觉）/ B（表格）之间选题
- 数据到手，要做对账、防泄漏、定结果文件格式
- 每一问要选模型、做实验、写评估
- 论文要按“首页摘要 + 第二页目录 + 正文 ≤30 页 + 附录全代码”组装
- 提交前要查赛道、CSV 格式、元数据、代码可运行
- 两个 AI（Codex / ZCode）要交接工作或恢复中断

## 三条铁律

1. **赛道必须选对。** 官方明文：赛道与内容不符**取消评奖资格**。选 A 还是 B 由数据形态与团队技术栈决定，不按兴趣。
2. **交付物三件套必须一致。** 论文数字 ↔ 结果文件 ↔ 源码/支撑材料。官方点名“程序跑不通”“结论不验证”是常见失分/违规点。
3. **先对账再建模。** 文件数、字段、类别分布、坐标口径、划分方式先与题面对齐；口径错一次，后面全部返工。

---

## 0. 开赛 10 分钟：填赛事参数

复制到 `00_赛题与规则/赛事参数.md`：

```markdown
P1 赛事与届次：        MathorCup 大数据竞赛（2026 第七届）
P2 赛道：              A（视觉）/ B（表格）——只能选一个
P3 作品语言/页数：      中文；正文 ≤30 页（附录不限）
P4 电子版第一页：       题目 + 摘要 + 关键词；第二页目录；正文页码从 1 开始
P5 附录与代码：         须含软件名称、命令、全部源程序
P6 支撑材料：           可选压缩包；结果文件按赛题要求
P7 匿名：               全文 + 元数据 + 代码路径不得出现学校/姓名/队号
（附加）提交：          论文 PDF（#****.pdf）+ 结果文件 + 承诺书
（附加）开赛：          2026-10-23 18:00（以官网当届通知为准）
```

**P2 的判据**：数据是图像/视频 → A；是表格/时序/运单/业务数据 → B。两条都像时，看团队是否有 GPU、是否会深度学习、是否有视觉经验。

---

## 1. 五阶段总览

| 阶段 | 时间（7 天赛程） | 核心产出 | 详细做法 |
|---|---|---|---|
| ① 读题 + 定赛道 | 第 1 天上午 | `选题结论.md`、`赛事参数.md` | `references/01_赛制与赛道选择.md` |
| ② 建工程 + 数据对账 | 第 1 天下午 | 目录骨架、`数据对账表.md`、结果模板 | `references/02_工程与数据契约.md` |
| ③ 逐问求解 | 第 2–5 天 | 每问：代码 + 指标 + 图表 + 实验记录 | `references/03_题型解法库.md`、`04_实验与评估.md` |
| ④ R 出图 + 论文 | 第 5–6 天 | 图件 + 论文 PDF | `references/05_出图规范_R.md`、`06_论文写作与摘要.md` |
| ⑤ 评估 + 提交 | 第 7 天 | 结果 CSV + 论文 + 支撑材料 + 自检 | `references/07_提交与红线.md` |

**压缩版（4 天）**：D1 读题+对账+baseline；D2 主力实验；D3 评估+出图；D4 论文+提交。论文从 D1 开始写，不要留到最后。

---

## 2. 题型 → 路线（速查）

| 题型 | 主线 | 必做对比 | 指标 |
|---|---|---|---|
| 图像分类 | 迁移学习 ResNet/DenseNet/EfficientNet/MobileNet | 3–4 个骨干 + 数据增强消融 | Acc/P/R/F1/AUC/混淆矩阵 |
| 目标检测/分割 | YOLO 系列（+注意力/多尺度）、Mask R-CNN、U-Net | 基线 vs 改进；分辨率对比 | mAP@0.5、mAP@0.5:0.95、IoU/Dice |
| 表格回归 | XGB/LGB/CatBoost + 特征工程 | 线性 vs 树模型 vs Stacking | R²/RMSE/MAE/MAPE |
| 表格分类 | XGB/LGB/RF/Stacking | 不平衡策略对比 | Acc/F1/AUC、约束满足率 |
| 时序预测 | ARIMA/SARIMA、指数平滑、Prophet、LSTM、集成 | 单模型 vs 加权集成 | MAPE/1-WMAPE/RMSE |
| 运筹优化 | 0-1/混合整数规划、贪心、SA、PSO/NSGA | 初解 vs 邻域改进；求解器对比 | 目标值、约束满足、Gap |
| 无监督/风险标注 | 分位数阈值、MQBL、孤立森林/OCSVM | 规则 vs 学习；约束满足 | 比例约束、密集度、下游 F1 |

**CV 题的固定三问**：Q1 分类（有没有）→ Q2 检测/分割（在哪、是什么）→ Q3 评估（多维度）。
**表格题的固定三问**：Q1 因素量化/打标 → Q2 预测 → Q3 不平衡或策略优化。

---

## 3. 数据对账（开赛 2 小时内必做）

```bash
# 文件数与结构
find data -type f | sed 's/.*\.//' | sort | uniq -c
# 图像：尺寸/模式/类别（CV）
python scripts/eda_yolo.py --images ... --labels ... --classes ...
# 表格：形状/缺失/目标分布（表格）
python -c "import pandas as pd; d=pd.read_csv('x.csv'); print(d.shape); print(d.isna().sum())"
```

对账表至少包含：文件数（对比题面）、字段与单位、类别分布、缺失率、异常值、划分方式、**泄漏风险**（同图多帧/同患者/同订单）。
本项目已知坑：2025 A 无负样本；2025 复赛按 ImageSets 划分；练习1 JSON 六列；练习2 png/jpg+RGB/RGBA/L 混合、同前缀多帧。

---

## 4. R 出图（首选；Python 保底）

**纪律：Python 算，R 画；接口是 CSV，不是内存。**

```r
source("scripts/plot_style.R")          # theme_mcup + 四类图表函数
p <- ggplot(df, aes(x, y)) + theme_mcup(legend = "br")
save_paper(p, "figures/fig1.png")       # 300dpi, ragg

plot_confusion_matrix(cm, c("dent","hole","rusty"), file="figures/cm.png")
plot_pr_curves(list(dent=list(recall=r,precision=p,ap=.804), all=list(...,ap=.810)), file="figures/pr.png")
plot_corr_heatmap(cor_mat, file="figures/corr.png")
plot_feature_importance(names, values, file="figures/imp.png")
```

关键数字（来自 615 张优秀论文图表实测）：白底、黑细框、**无网格**、图例内嵌无边框、3:2；中文宋体 + 拉丁 Times；300dpi。
配色：训练曲线四色（`#1f77b4/#ff7f0e/#2ca02c/#d62728`）；混淆矩阵 Blues；热力图 hot；PR 总曲线深蓝加粗；特征重要性逐条冷暖色。
**不要混用第二个绘图后端**；R 不可用时才用 `plot_style.py` 的 `set_style("m杯")`。

---

## 5. 论文（30 页硬约束）

```
第1页   题目 + 摘要 + 关键词
第2页   目录
正文    一、问题重述 / 二、问题分析 / 三、模型假设 / 四、符号说明
        五、数据预处理 / 六、七、八、每问模型建立与求解 / 九、模型评估 / 十、模型评价与推广
参考文献（单独起页）
附录（页数不限，含软件名、命令、全部源码）
```

- **摘要四段式**：背景 1–2 句 + 总述方法 → 针对问题一/二/三各一段（方法 + 关键数字）→ 关键词。
- 每问正文按“任务本质 → 难点 2–3 条 → 对策 → 模型 → 结果 → 验证”组织。
- 图表必须突出结论与比较；每图/表在正文被引用；图题在下、表题在上。
- 官方明文：无页眉、无身份信息、中文、正文 ≤30 页；附录给全部源代码与运行命令。

---

## 6. 官方六大答卷问题（提交前逐条自查）

官方讲座原文：
> 数据预处理没有章法，不按照需求去处理；问题理解不明确（细节要自己确定方案，现场只提供需求），模型条件约简不符合实际，只注重展示方法，不注重问题解决；结论不验证实际，不讨论；程序跑不通。

| # | 自查项 | 验证 |
|---|---|---|
| 1 | 清洗步骤表（每步行数变化+理由），按题目需求处理 | 论文预处理节列 5–8 步且有数字 |
| 2 | 每问写明边界、输出格式、自定假设 | 问题分析每问一段“本题要求回答……” |
| 3 | 假设可辩护，简化说明代价 | 每条假设后写“若放宽会怎样” |
| 4 | 方法与问题一一对应，不堆模型 | 每个模型能回答“解决哪条难点” |
| 5 | 有误差分析/失败案例/业务解释 | 每问结果后一段“合理性/局限” |
| 6 | 干净环境可复现，输出与论文一致 | 提交前重跑一次全流程 |

完整清单见 `assets/checklists.md`。

---

## 7. 结果文件与校验

| 题型 | 模板 | 字段 |
|---|---|---|
| 分类 | `assets/result_templates/result_classification.csv` | `image_id,prediction,probability` |
| 检测 | `assets/result_templates/result_detection.csv` | `image_id,class_id,x_center,y_center,width,height`（0–1 归一化） |
| 分配/优化 | `assets/result_templates/result_assignment.txt` | `id,serviceStartTime,auntId` |
| 分批决策 | `assets/result_templates/result_assignment_batch.txt` | `currentTime,id,serviceStartTime,auntId,retainable` |

提交前：

```bash
python scripts/audit_mcup_submission.py --root . --paper 05_论文/论文.pdf \
  --result 03_结果/result_detection.csv --images-dir 01_数据/test
```

它会检查：论文页数与元数据、结果表头/坐标范围/类别 id/覆盖、必交文件是否齐全。

---

## 8. 三方协作（Codex / ZCode / dsh）

| 角色 | 位置 | 职责 | 边界 |
|---|---|---|---|
| **Codex** | WSL | **核心工作**：建模推理、模型选型、结果解读、代码/训练/评估、论文 LaTeX | 最终技术主张由 Codex 负责 |
| **ZCode** | Windows | 辅助：环境、数据、文档、复核、提交检查 | 能力与 dsh 相近，但不代做 dsh 的评审 |
| **dsh（DeepSeek Harness）** | Windows | **论文评审与优化**：结构/逻辑/表达/格式/一致性审查，给出修改建议 | 不承担建模推理、选型、结果解读 |

- 目录：`docs/collab/`，单一写入者：`to_codex.md` 只有 ZCode 写、`to_zcode.md` 只有 Codex 写，收件方只改状态行。
- 状态机：`OPEN → ACK → DONE / BLOCKED / PAUSED`；**静默模式**：只在交接/阻塞/决策时发消息。
- dsh 的评审写入 `docs/collab/reviews/REV-*.md`（六维 rubric + P0/P1/P2 建议），由 ZCode 转成 task 消息交 Codex 落实；dsh 不改代码/结果/建模主张。
- 提交：`collab(MSG-ID): ...`；开工先 `git pull --ff-only`，处理完立即推送。

细节见 `references/08_协作与恢复.md`。

---

## 9. Top 10 MathorCup 常见错误

| # | 错误 | 后果 | 防法 |
|---|---|---|---|
| 1 | 赛道与内容不符 | **取消评奖资格** | 开赛 10 分钟内定赛道，写进赛事参数 |
| 2 | 结果 CSV 坐标用像素而非 0–1 | 提交无效 | `audit_mcup_submission.py` 校验 |
| 3 | 无负样本却强行训练二分类 | 分类器学不到“无残损” | 用标注框遮挡/裁剪构造负样本（2025-1/2025-3 做法） |
| 4 | 同图多帧/同患者跨训练验证 | 指标虚高、结论不可信 | 按文件名前缀 GroupShuffleSplit |
| 5 | 摘要数字与正文/表格不一致 | 评审直接扣分 | 摘要每个数字回正文核对 |
| 6 | 只报准确率，忽略召回/AP | 任务指标错配 | 分类看 F1/召回，检测看 mAP，分割看 Dice/IoU |
| 7 | 无消融/无对比 | 方法可信度不足 | 至少“基线 vs 完整”；逐模块加 |
| 8 | 论文超 30 页/有页眉/身份信息 | 格式违规 | 提交前逐项检查，清 PDF 元数据 |
| 9 | 附录代码缺 import/路径硬编码 | 程序跑不通（违规） | 干净环境按 README 重跑 |
| 10 | 结果文件多版本并存 | 交错提交 | 只保留最终版，旧版进归档 |

---

## 10. 停止条件

1. 边际收益 < 成本（一处 0.3 分要 2 小时）
2. 只剩结构性大改（会引发全局重排）
3. 剩余 < 3 小时：停止改内容，只做提交检查
4. 连续 2 轮评估无提升：检查口径而非继续调参

---

## 11. 参考文件索引

| 文件 | 什么时候读 |
|---|---|
| `references/01_赛制与赛道选择.md` | 开赛 0–1 小时 |
| `references/02_工程与数据契约.md` | 建工程、第一轮数据对账 |
| `references/03_题型解法库.md` | 每问开工 |
| `references/04_实验与评估.md` | 做对比/消融/评估章节 |
| `references/05_出图规范_R.md` | 第一次出图前 |
| `references/06_论文写作与摘要.md` | 论文组装、写摘要 |
| `references/07_提交与红线.md` | 提交前 3 小时 |
| `references/08_协作与恢复.md` | 分工、换 AI、中断恢复 |
| `references/09_案例库与官方口径.md` | 开局通读，找同类题 |
| `references/10_论文评审与优化.md` | dsh 评审时（六维 rubric、P0/P1/P2、输出模板） |
| `references/11_自强平台操作指南.md` | 上 A100 平台前（容器/GPU/同步/长任务/报错） |
| `references/12_AI工具合规.md` | 论文声明/参考文献/标注/支撑材料 PDF（2026 新规） |
| `assets/project_tree.txt` | 建目录时对照 |
| `assets/paper_outline.md` | 论文骨架 |
| `assets/r_theme_mcup.R` | 出图主题 |
| `assets/checklists.md` | 六大问题 + 提交 20 项 + 图表 QA |
| `assets/result_templates/` | 结果文件模板 |
| `scripts/new_mcup_project.py` | 一键建骨架 |
| `scripts/audit_mcup_submission.py` | 提交前自检 |
| `scripts/collab_msg.py` | 双 AI 消息 |
