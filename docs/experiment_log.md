# 实验记录

| 实验 ID | 问题 | 状态 | 配置 | 产物 | 结论 |
|---|---|---|---|---|---|
| E000 | Q1 | FROZEN_RECHECK | `02_代码/Q1_感官评分/configs/E000_data_contract.yaml` | `03_结果/Q1_感官评分/q1_data_contract.json`、`01_数据/compound_name_map.csv`、目标字典、附件 6 行映射、固定 CV | 旧数据产物；官方修正后先做附件 diff 并重跑数据契约 |
| E001 | Q1 | FROZEN_RECHECK | `02_代码/Q1_感官评分/configs/E001_baselines.yaml` | `03_结果/Q1_感官评分/q1_*_E001.*` | 旧数据结果，待官方修正后复核，不得用于当前结论 |
| E002 | Q1 | FROZEN_RECHECK | `02_代码/Q1_感官评分/configs/E002_robust_linear.yaml` | `03_结果/Q1_感官评分/q1_*_E002.*` | 旧数据结果，待官方修正后复核，不得用于当前结论 |
| E003 | Q1 | FROZEN_RECHECK | `02_代码/Q1_感官评分/configs/E003_extra_trees.yaml` | `03_结果/Q1_感官评分/q1_*_E003.*`、`figure_data/` | 旧数据结果，待官方修正后复核；关联 7 张图稿也不验收、不定稿 |

每次运行追加新行，不覆盖旧实验。指标必须链接到机器生成文件，论文数字以结果目录中的冻结版本为准。

