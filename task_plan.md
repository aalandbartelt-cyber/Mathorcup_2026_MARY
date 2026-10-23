# Task Plan: 2026 初赛赛道首判

## Goal
在不训练、不占用 GPU 的前提下，于 2026-10-23 19:00 前完成 A/B 题原文核验、附件盘点、赛道建议、两条技术路线和今晚六小时计划。

## Current Phase
Complete

## Phases

### Phase 1: 题面与交付要求
- [x] 提取并核验 A/B 题 PDF、A 数据说明、AI 合规规定
- [x] 建立逐问任务—输入—输出—约束表
- **Status:** complete

### Phase 2: 数据盘点
- [x] 统计 A 各表形状、列名、缺失、目标与变换口径
- [x] 列出 B 压缩包内容、数据格式、时间范围与缺失线索
- **Status:** complete

### Phase 3: 赛道与路线决策
- [x] 对比数据完备性、方法风险、算力与外部依赖
- [x] 给出两条完整路线、首选和验证矩阵
- **Status:** complete

### Phase 4: 时间盒与交付
- [x] 制定至 24:00 的六小时计划
- [x] 核对缺数据、权限和协作事项
- [x] 形成第一版判断文档
- **Status:** complete

## Key Questions
1. A 的样本—标签—化合物—风格映射是否足以支撑三问并形成共享主线？
2. B 的时空数据是否无需外部数据即可形成可验证的训练/回测闭环？
3. 哪个赛道在七天内更容易产出可复现、可解释、数字一致的论文与结果？

## Decisions Made
| Decision | Rationale |
|---|---|
| 不运行训练或 GPU 模型 | 用户明确要求先盘数据、定赛道 |
| 先检查压缩包目录，不先展开 257MB RAR | 控制时间与磁盘；先判断数据形态和依赖 |
| 推荐锁定 A 赛道 | A 有完整的 250×231 预测特征与 250×50 目标闭环；B 缺少可直接回测的事件级标签且仿真依赖重 |
| A 首选稳健统计路线 | n=250 且风格长尾，PLS2/MultiTaskElasticNet + 条件 copula 比直接深生成更可验证 |

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| `pdftotext`/`pdfinfo` 未安装 | 1 | 改用环境内 Python PDF 库；若文本层缺失再截图核对 |
| SciPy 无法读取附件 4 的 NetCDF4/HDF5 | 1 | 记录为 B 环境缺口；未安装依赖，保持赛前只读检查 |

