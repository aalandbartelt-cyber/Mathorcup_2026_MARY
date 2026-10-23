# 实验记录

| 实验 ID | 问题 | 状态 | 配置 | 产物 | 结论 |
|---|---|---|---|---|---|
| E000 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E000_data_contract.yaml` | `q1_data_contract.json`、`q1_preprocessing_steps.csv`、`q1_feature_scale_audit.csv`、`q1_missingness_by_feature.csv`、`01_数据/compound_name_map.csv` | 已按负责老师说明固化实际数据清洗口径：beer主键、37列保持log10、73个缺失折内处理、全样本保留、目标为连续标准化值 |
| E001 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E001_baselines.yaml` | `03_结果/Q1_感官评分/q1_*_E001.*` | 实现与新口径一致：折内中位数+缺失指示、无删样/二次取对数；输入未变，无需重跑 |
| E002 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E002_robust_linear.yaml` | `03_结果/Q1_感官评分/q1_*_E002.*` | 实现与新口径一致；秩高斯为折内模型变换而非数据清洗，输入未变，无需重跑 |
| E003 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E003_extra_trees.yaml` | `03_结果/Q1_感官评分/q1_*_E003.*`、`figure_data/` | 实现与新口径一致：树模型保留附件尺度、折内中位数+缺失指示、无删样；指标继续有效 |
| E100 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E100_contract.yaml` | `03_结果/Q2_关键化合物/q2_candidate_dictionary.csv`、`q2_graph_*` | 候选、名称和有符号 Spearman 图合同通过；218 个严格化合物候选 |
| E101–E104 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E101_E104_selection.yaml` | `03_结果/Q2_关键化合物/q2_*_E101_E104.*` | 10×3 OOF 完成；E103 宏 MAE 0.3529，E104 0.3530；图惩罚未得到预测增益，扩展参数敏感性进入 E109 |
| E109 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E109_boundary_sensitivity.yaml` | `03_结果/Q2_关键化合物/q2_e109_*` | 扩展网格下 E103 的指标与 Top-3 完全不变；E104 未改善 MAE，图惩罚降为消融组 |
| E105 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E105_extra_trees_permutation.yaml` | `03_结果/Q2_关键化合物/q2_*_E105.*` | ExtraTrees 宏 $R^2$/MAE=0.2359/0.3422，10/10 次重复优于 E103；预测主线晋级，字段 Top-3 待共识与稳定性审计 |
| E106 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E106_rank_consensus.yaml` | `03_结果/Q2_关键化合物/q2_*_E106.*` | 生成 50×3 三方法统计共识；S1/S2/S3=41/90/19，PLS 敏感性 Jaccard=0.652，尚需 E107/E108 |

每次运行追加新行，不覆盖旧实验。指标必须链接到机器生成文件，论文数字以结果目录中的冻结版本为准。

