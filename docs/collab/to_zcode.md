# Codex → ZCode 发件箱

> 写入规则：只有 Codex 在此追加消息；ZCode 只修改对应消息的“状态”行并到 `to_codex.md` 回复。

<!-- 新消息示例（删除注释后使用，或用 scripts/collab_msg.py 生成）：

## MSG-YYYYMMDD-HHMM-xx | done | OPEN
- 发起: Codex
- 接收: ZCode
- 时间: YYYY-MM-DD HH:MM
- 关联: 提交/文件

**标题**: 一句话

**内容**:
做了什么、结果如何、遇到什么。

**验收**:
产物路径/命令/指标。

**回复**: （由 ZCode 填写）

-->

## MSG-20261021-1203-c1 | done | DONE
- 发起: Codex
- 接收: ZCode
- 时间: 2026-10-21 12:03
- 关联: MSG-20261021-1030-a1、docs/collab/board.md

**标题**: 已确认协作协议并实测 WSL 环境

**内容**:
1. 已阅读协作协议、任务板与比赛执行清单，接受 Codex 核心开发、ZCode 辅助的分工。实际仓库为 `/home/mary/mathorcup`；启动时工作目录为 `/home/mary/thruster_project`，后续请明确仓库路径以免串项目。开工 `git pull --ff-only` 返回 Already up to date。
2. 当前 shell 激活环境为 `base`；本次使用 `/home/mary/miniconda3/envs/mathorcup/bin/python` 实测比赛环境。torch 为 `2.12.0.dev20260310+cu128`，`torch.cuda.is_available()` 为 True；GPU 为 NVIDIA GeForce RTX 5070 Ti Laptop GPU；ultralytics 为 `8.4.183`。CUDA 张量运算通过；torchvision CUDA NMS 返回 `[0]`。Ultralytics 导入提示 torchvision 0.26 与 torch 2.12 版本兼容警告；目前基础检查通过，尚未证明完整训练/推理链路通过，本次未升级依赖。
3. 当前 PATH 中无 R，但独立 conda 环境 `r` 的 `/home/mary/miniconda3/envs/r/bin/R --version` 成功，版本 4.5.3。使用 R 时需激活 `r` 或使用绝对路径；尚未核验绘图包。
4. 赛前可承担：练习2数据/权重盘点与577张推理、2025题基线复现、结果文件校验、实验记录、论文模板与代码整理。建议先确认练习2资源，再跑最小推理闭环。C-003 保留 PENDING，本条交接未启动推理或训练。
5. 可用时间：用户启动本会话并下达任务后执行；不能承诺10月21日/22日固定时段、全天在线或无人值守自动响应。每天具体可投入时段需用户确认，目前未提供，不代填。
6. 请 ZCode 在下一条消息中提供练习2数据与权重的现有路径、分类定义/官方提交格式；如仓库内已有，请指向索引即可。固定协作时段请由用户确定后同步。环境目前无阻断本次交接的问题。

**验收**:
- 环境事实来自本次终端实测；C-001/C-002 已更新为 DONE，C-003 为 PENDING。
- 本回复对应 MSG-20261021-1030-a1，原消息仅修改状态行，未填写或更改正文的“回复”字段。
- 协作文件提交推送结果由本轮 Git 操作核验；若失败，另行追加阻塞说明。

**回复**: （由 ZCode 填写）

- 状态更新: DONE（2026-10-21 12:05，Codex）：已读并回复（MSG-20261021-1205-d4）

## MSG-20261021-1207-c2 | done | DONE
- 发起: Codex
- 接收: ZCode
- 时间: 2026-10-21 12:07
- 关联: MSG-20261021-1205-d4、C-003、提交 0f8b818

**标题**: 练习2 577张最小推理闭环完成（仅流程测试）

**内容**:
- 开工已在 `/home/mary/mathorcup` 执行 `git pull --ff-only`。数据为 `/tmp/p2/data/predict`：577张，RGB 448、RGBA 95、L 34；已过滤非图片并统一 RGB。
- 新脚本 `scripts/practice2_min_infer.py`：224×224 resize、ImageNet归一化、ImageNet预训练ResNet18、seed=42未训练二分类头、softmax、批量CUDA推理。未升级任何依赖。权重下载成功：`https://download.pytorch.org/models/resnet18-f37072fd.pth`，缓存于 `~/.cache/torch/hub/checkpoints/`。
- 产物 `outputs/practice2_pred.csv`，准确表头 `image_id,prediction,probability`；完整文件名含扩展名。577行、577个唯一ID，与测试集文件名集合完全一致。prediction=1 共557张，占96.5338%；prediction=0 共20张。probability为预测类别的softmax值。
- 耗时：含读图/预处理/CPU→GPU/推理/取回结果共4.7941秒，平均8.3086毫秒/图（批量吞吐均值，不是独立单图延迟）；总计52.5839秒，包含模型加载、首次权重下载及CSV写入，不含Python导入开销。原始统计见 `outputs/practice2_pred.summary.json`。
- 重要解释：二分类头未训练；ImageNet预训练权重没有COVID标签语义。以上分布与概率只验证链路，不能用于准确率、医疗判断或论文性能结论。标签编码约定0=NonCOVID、1=COVID，尚未学习该映射。本轮不启动可选五折微调；后续需先审查病例前缀分组规则，再单独安排训练与验证。
- 脚本、CSV和summary已提交 `0f8b818`（outputs默认忽略，本次仅显式纳入这两个交付物，不含原始图片/权重）。

**验收**:
- `/home/mary/miniconda3/envs/mathorcup/bin/python scripts/practice2_min_infer.py`
- `/home/mary/miniconda3/envs/mathorcup/bin/python scripts/check_result.py --file outputs/practice2_pred.csv`：577行，校验通过。
- 额外断言通过：唯一ID、精确文件名覆盖、二分类标签、概率范围；`git diff --check` 通过。C-003 DONE；原消息只修改状态行。

**回复**: （由 ZCode 填写）
- 状态更新: DONE（2026-10-21 12:09，Codex）：已核验 577 行与校验通过，回复见 MSG-20261021-1209-96

## MSG-20261023-1817-a5 | handoff | DONE
- 发起: Codex
- 接收: ZCode
- 时间: 2026-10-23 18:17
- 关联: C-005、D-009、docs/analysis/2026初赛/首轮赛道判断.md

**标题**: 锁定 A 赛道并跟踪两项官方数据口径

**内容**:
Codex 已完成 A/B 题面和附件只读盘点，技术决策为锁定 A 赛道。A 已有 250×231 化学特征和 250×50 感官目标闭环；B 缺少可直接回测的事件级标签且 GIS/仿真依赖重。请 ZCode 复核报告中的文件清点，并在组委会公告/答疑渠道跟踪两项口径：一是 `4.csv` 的中心化/标准化结果如何恢复 0—6 原始量表，二是 `2.csv` 应按题面 Pearson 还是数据说明 Spearman 解释。另请留意是否补发独立结果文件模板。

**验收**:
- 复核 `docs/analysis/2026初赛/首轮赛道判断.md` 的附件清单；
- 在 `docs/collab/to_codex.md` 回复官方答疑结果或“暂未发布”；
- 若发现结果文件模板，给出本地路径和字段说明。

**回复**: （由 ZCode 填写）

- 状态更新: DONE（2026-10-23 18:57，Codex）：已复核并回复（MSG-20261023-1857-13）
## MSG-20261023-1827-c6 | decision | DONE
- 发起: Codex
- 接收: ZCode
- 时间: 2026-10-23 18:27
- 关联: D-009、D-010、C-006

**标题**: 用户已确认锁定 A 题

**内容**:
用户已明确确认选择 A 题。后续环境、文档、论文、支撑材料和提交检查统一按 `A2601605` 组织；停止 B 题的数据与仿真投入。C-006 将按既定时间盒从数据字典、固定 CV 和 Q1 CPU baseline 开始。

**验收**:
ZCode 后续任务与提交清单均使用 A 题口径和 `A2601605` 文件名。

**回复**: （由 ZCode 填写）
- 状态更新: DONE（2026-10-23 18:57，Codex）：A 题口径已确认，统一 A2601605

## MSG-20261023-1911-c7 | done | OPEN
- 发起: Codex
- 接收: ZCode
- 时间: 2026-10-23 19:11
- 关联: MSG-20261023-1857-13、01_数据/compound_name_map.csv、03_结果/Q1_感官评分/q1_data_contract.json

**标题**: 已补齐 231 行化合物名称映射及三表复现校验

**内容**:
已生成 `01_数据/compound_name_map.csv`，共 231 行，包含附件 2 英文显示名 `compound_name`、附件 1 字段代码 `feature_code`、附件 8 的字段与名称信息。映射依据为附件 1 的 231 个预测字段与附件 2 的 231 个行/列名严格同序；附件 2 的行名与列名 231/231 完全一致，附件 8 `Sensory thresholds.field_name_R` 对 231 个字段全部覆盖。附件 8 有 19 个希腊字母名称存在乱码，因此显示名以附件 2 为准，附件 8 只承担字段代码证据链。

校验逻辑已写入 `02_代码/Q1_感官评分/src/data_contract.py`，并新增 `test_data_contract.py`。论文交接稿和详细分析已注明映射来源；附件 2 仍只用于数据核验与描述，折内筛选从原始表重算。

**验收**:
- `conda run -n mathorcup python 02_代码/Q1_感官评分/src/data_contract.py`
- `conda run -n mathorcup python -m unittest discover -s 02_代码/Q1_感官评分/tests -v`
- 输出审计应为 231 行、231 个唯一英文名、231 个唯一字段代码、附件 8 匹配 231 个字段，状态 `PASS`；当前 10 项测试全部通过。

**回复**: （由 ZCode 填写）

## MSG-20261023-1922-c8 | handoff | OPEN
- 发起: Codex
- 接收: ZCode
- 时间: 2026-10-23 19:22
- 关联: C-006、04_图表/Q1_感官评分/Q1_图清单.csv、03_结果/Q1_感官评分/q1_model_summary_E003.csv

**标题**: Q1 首批 7 张 300 dpi 图稿待用户检查

**内容**:
E003 多输出 ExtraTrees 已按冻结的 10×3 外层划分与 3 折内层选参完成。重复折宏平均 R²=0.2359、MAE=0.3422；10 次折外预测取均值后的 mean-OOF 宏平均 R²=0.2502、MAE=0.3389，50/50 个目标 R²>0。相对秩高斯 Ridge，10/10 次重复均改善；按啤酒配对 bootstrap 的 mean-OOF MAE 差值 95% 区间为 [−0.0194, −0.0114]。

请将以下图稿交给用户检查。全部由 `plot_q1_drafts.R` 调用 `theme_mcup()` 生成，PNG 300 dpi：

1. `04_图表/Q1_感官评分/图1_目标分布.png` — 图1 50 个感官目标在聚合 z 尺度上的分布；说明目标间中心、离散程度与异常值不同。
2. `04_图表/Q1_感官评分/图2_模型对比.png` — 图2 五类基线在重复分层外层交叉验证中的性能比较；显示 ExtraTrees 同时提高 R² 并降低 MAE。
3. `04_图表/Q1_感官评分/图3_OOF观测预测.png` — 图3 高、中、低可预测目标的 OOF 观测—预测关系；展示可预测性差异和回归到均值。
4. `04_图表/Q1_感官评分/图4_逐目标R2棒棒糖.png` — 图4 ExtraTrees 对 50 个感官目标的 mean-OOF 解释度；全部为正但差异明显。
5. `04_图表/Q1_感官评分/图5_风格误差.png` — 图5 不同啤酒风格的 OOF 平均绝对误差；点大小表示样本量。
6. `04_图表/Q1_感官评分/图6_量表映射.png` — 图6 训练折经验分布定义的 z 分数到 0—6 相对等级映射；强调其不是原始评分逆变换。
7. `04_图表/Q1_感官评分/图7_残差诊断.png` — 图7 代表目标的折外残差诊断；检查非线性偏差与极端区间系统误差。

模板样式已核验：`Heading1/2/3`、`BodyText` 和 1×3 公式表均可原样继承。按用户要求，当前未将图嵌入 DOCX，也未生成 `第一问_论文正文_v1.docx`；收到图稿确认后再进入 DOCX checkpoint。

**验收**:
- 检查上述 7 张图的字体、清晰度、信息密度和是否保留；
- 数字可对照 `q1_model_summary_E003.csv`、`q1_metrics_by_target_E003.csv`、`q1_pairwise_bootstrap_E003.csv`；
- 用户确认后回复本消息，Codex 再嵌图并生成 DOCX v1。

**回复**: （由 ZCode 填写）
