# 实验记录

| 实验 ID | 问题 | 状态 | 配置 | 产物 | 结论 |
|---|---|---|---|---|---|
| E000 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E000_data_contract.yaml` | `q1_data_contract.json`、`q1_preprocessing_steps.csv`、`q1_feature_scale_audit.csv`、`q1_missingness_by_feature.csv`、`01_数据/compound_name_map.csv` | 已按负责老师说明固化实际数据清洗口径：beer主键、37列保持log10、73个缺失折内处理、全样本保留、目标为连续标准化值 |
| E001 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E001_baselines.yaml` | `03_结果/Q1_感官评分/q1_*_E001.*` | 实现与新口径一致：折内中位数+缺失指示、无删样/二次取对数；输入未变，无需重跑 |
| E002 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E002_robust_linear.yaml` | `03_结果/Q1_感官评分/q1_*_E002.*` | 实现与新口径一致；秩高斯为折内模型变换而非数据清洗，输入未变，无需重跑 |
| E003 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E003_extra_trees.yaml` | `03_结果/Q1_感官评分/q1_*_E003.*`、`figure_data/` | 实现与新口径一致：树模型保留附件尺度、折内中位数+缺失指示、无删样；指标继续有效 |

每次运行追加新行，不覆盖旧实验。指标必须链接到机器生成文件，论文数字以结果目录中的冻结版本为准。

