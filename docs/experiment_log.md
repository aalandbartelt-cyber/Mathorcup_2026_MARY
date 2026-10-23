# 实验记录

| 实验 ID | 问题 | 状态 | 配置 | 产物 | 结论 |
|---|---|---|---|---|---|
| E000 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E000_data_contract.yaml` | `03_结果/Q1_感官评分/q1_data_contract.json`、`01_数据/compound_name_map.csv`、目标字典、附件 6 行映射、固定 CV | 250×231→250×50；231 行名称映射三表校验通过；附件 2/5/6 均为 Spearman；末尾感官列存在顺序差异 |
| E001 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E001_baselines.yaml` | `03_结果/Q1_感官评分/q1_*_E001.*` | 原尺度线性模型受极端折外值影响；收缩风格基线重复折宏平均 MAE 0.3579 |
| E002 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E002_robust_linear.yaml` | `03_结果/Q1_感官评分/q1_*_E002.*` | 折内秩高斯修复异常外推；Ridge 重复折宏平均 R² 0.2001、MAE 0.3566，PLS2 为 0.1823、0.3637 |
| E003 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E003_extra_trees.yaml` | `03_结果/Q1_感官评分/q1_*_E003.*`、`figure_data/` | 多输出 ExtraTrees 重复折宏平均 R² 0.2359、MAE 0.3422；10 次预测均值的 mean-OOF R² 0.2502、MAE 0.3389，50/50 个目标 R²>0 |

每次运行追加新行，不覆盖旧实验。指标必须链接到机器生成文件，论文数字以结果目录中的冻结版本为准。

