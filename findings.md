# Findings & Decisions

## Requirements
- 19:00 前给出赛道建议、A/B 数据盘点、两条路线、今晚计划、协作事项和风险。
- 阅读两份赛题 PDF、A 数据说明及公共 AI 合规规定。
- 不训练、不占 GPU；R 负责最终出图，Python/sklearn/XGB/LGB 负责建模。

## Research Findings
- A 附件为 6 个 CSV、2 个 XLSX 与数据说明，总体约 4.3MB。
- B 附件为 7 个 ZIP 与 1 个 RAR；RAR 约 257MB，B 附件总体约 284MB。
- 当前时间为 2026-10-23 18:06 CST。
- A 数据说明：1=250 个样品化学量；2/5/6=化学内部、感官内部、化学—感官 Spearman 相关；3=品评表；4=感官结果；7=论坛描述符；8=其他数据。
- A `1.csv` 为 250×234（231 个化学特征），仅 73 个缺失单元；`4.csv` 为 250×53（50 个感官目标），无缺失。
- `3.xlsx` 实际有 12 张品评表和 summary，共定义 50 个目标；summary 全零，是 0—6 空白模板。
- 对照附件 8 原量纲表，37 个字段精确满足 `raw=10^x`，33 个字段保持原值，没有字段支持自然对数；所以对数底数为 10。
- 题面称 `2.csv` 为 Pearson，数据说明称 Spearman；应从 `1.csv` 重算 Spearman 并对照附件。
- `4.csv`/`7.csv` 的 `beer_id` 精度损失，仅 96 个唯一值；跨表必须用 `beer`。
- B 小时数据覆盖 2021—2026，MET/RADM/SOIL/FLUX 各 52,584 行；FLUX 约 24% 测量值缺失，其他表有 11%—13% 非零 QC 标志。
- B 的 2026 小时表延伸到 12 月，预报建模必须按起报时点截断；每日气温只到 2025-08-02。
- B 历史灾点仅 84 条，缺少已配对的逐事件触发时刻和气象序列；Q3 覆盖率和 Q4 仿真均存在明显数据缺口。

## Technical Decisions
| Decision | Rationale |
|---|---|
| 推荐锁定 A 赛道 | A 的样本—化学—感官—风格链路完整；B 的事件级验证闭环不足且地学软件依赖未就绪 |
| A 采用稳健统计主线、非线性增强备选 | n=250、p=231、50 目标且重点风格仅 10—28 个样本，先保证 OOF 稳定性 |

## Issues Encountered
| Issue | Resolution |
|---|---|
| 系统无 `pdftotext`/`pdfinfo` | 使用 Python PDF 库读取文本；必要时截图核对扫描页 |
| 当前无 GDAL/rasterio/netCDF4 | 已用 ZIP/RAR 清单、Pillow GeoTIFF 标签和 DBF 头完成只读盘点；列为 B 赛道依赖风险 |

## Resources
- `data/2026初赛/`
- `~/.codex/skills/mary-modeling-contest-pipeline-mcup/`
- `docs/analysis/2026初赛/首轮赛道判断.md`

## Visual/Browser Findings
- 尚未进行页面截图；题面将以 PDF 文本提取为主，必要时截图核对。

