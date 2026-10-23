# Task Plan: A 题第一问感官评分

## Goal
建立 A 题第一问从 231 维化学数据到 50 维感官结果的无泄漏预测闭环，给出可复现的 0—6 相对等级、逐目标性能、不确定性与论文证据链。

## Current Phase
Q1 data contract and CV frozen; leakage-safe CPU baseline next

## Phases

### Phase 1: 第一问题意与尺度识别
- [x] 核验附件 1/3/4/8 的样本、字段、量表和目标定义
- [x] 追溯原始论文、补充数据与建模代码
- [x] 证明附件 4 不能唯一逆变换为原始 0—6 评分
- **Status:** complete

### Phase 2: 数据诊断与实验预注册
- [x] 量化输入缺失、共线、低维结构和化学锚点
- [x] 量化目标偏度、相关结构、风格效应和历史可预测性
- [x] 固化主验证、压力测试、指标、模型矩阵与消融
- **Status:** complete

### Phase 3: 数据字典与无泄漏管线
- [x] 生成 50 目标字典、字段分组和固定 CV split
- [ ] 实现折内填补、缩放、缺失指示和双尺度输出
- [ ] 加入数据键、列数、取值和泄漏断言
- **Status:** in progress

### Phase 4: CPU baseline 与增强模型
- [ ] E0 全局均值、E1 收缩风格均值、E2 Ridge、E3 PLS2
- [ ] ExtraTrees/Gradient Boosting 小网格与低秩模型
- [ ] 生成重复 OOF、逐目标指标和模型选择结果
- **Status:** pending

### Phase 5: 校准、图表与论文结果
- [ ] 经验分位数与稳健仿射 0—6 映射敏感性
- [ ] 共形预测区间和留一风格压力测试
- [ ] 用 R/ggplot2 `theme_mcup()` 生成 Q1 图表
- [ ] 写入论文方法、结果、合理性和局限
- **Status:** pending

## Key Questions
1. 化学特征在严格折外验证下能可靠预测哪些感官目标？
2. 多任务共享结构和逐目标树模型如何组合而不产生选择偏差？
3. 在缺少评委层面逆变换参数时，如何诚实、可复现地给出 0—6 分级？
4. 模型在稀有风格和未见风格上是否仍可靠？

## Decisions Made
| Decision | Rationale |
|---|---|
| 用 `beer` 连接、禁用 `beer_id` | 附件 4 的 `beer_id` 精度损失，仅 96 个唯一值 |
| 主任务预测附件 4 的 z 分数 | 这是唯一完整监督标签；原始 0—6 无法唯一恢复 |
| 同时输出 z 预测与 0—6 相对等级 | 既保留可评价尺度，也满足题面分级表达 |
| 主验证用 10×3 重复风格分层 CV | Faro 只有 3 个样本，五折无法分层 |
| 化学-only 为主，风格作基线/消融 | 防止风格查表替代化学—感官学习 |
| 低秩模型 + 逐目标树模型 | 目标共享结构强，但历史上不同目标的最优模型明显不同 |
| 相关分析统一用 Spearman | 附件 2/5/6 已逐元素复算确认；Pearson 仅用于线性与校准诊断 |
| 新增编号目录并按问题对齐 | 每问均有代码、结果、图表和论文交接入口；旧路径仅为兼容保留 |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| `pdftotext`/`pdfinfo` 未安装 | 1 | 使用环境内 PyMuPDF 提取题面文本 |
| 默认 Python 无 pandas | 1 | 使用 `conda run -n mathorcup python` 执行分析 |
| `conda run` 通过 stdin 未输出 | 1 | 改用临时脚本或 `python -c` |

