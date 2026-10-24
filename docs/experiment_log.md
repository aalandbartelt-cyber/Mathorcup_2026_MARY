# 实验记录

| 实验 ID | 问题 | 状态 | 配置 | 产物 | 结论 |
|---|---|---|---|---|---|
| E000 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E000_data_contract.yaml` | `q1_data_contract.json`、`q1_preprocessing_steps.csv`、`q1_feature_scale_audit.csv`、`q1_missingness_by_feature.csv`、`01_数据/compound_name_map.csv` | 已按负责老师说明固化实际数据清洗口径：beer主键、37列保持log10、73个缺失折内处理、全样本保留、目标为连续标准化值 |
| E001 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E001_baselines.yaml` | `03_结果/Q1_感官评分/q1_*_E001.*` | 实现与新口径一致：折内中位数+缺失指示、无删样/二次取对数；输入未变，无需重跑 |
| E002 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E002_robust_linear.yaml` | `03_结果/Q1_感官评分/q1_*_E002.*` | 实现与新口径一致；秩高斯为折内模型变换而非数据清洗，输入未变，无需重跑 |
| E003 | Q1 | COMPLETE | `02_代码/Q1_感官评分/configs/E003_extra_trees.yaml` | `03_结果/Q1_感官评分/q1_*_E003.*`、`figure_data/` | 实现与新口径一致：树模型保留附件尺度、折内中位数+缺失指示、无删样；指标继续有效 |
| E004 | Q1 | SCREENED_NOT_PROMOTED | `02_代码/Q1_感官评分/configs/E004_ordinal_hgb.yaml` | `03_结果/Q1_感官评分/q1_*_E004_screening.*`、`q1_screening_*_E004.*` | 与 E003 同一 3 次筛查重复下，仅 MAE 改善（0.3365 vs 0.3429）；R²/Spearman/代理 QWK/代理 ACC±1均下降，按冻结预门槛不扩展至 10 次 |
| E004_GPU | Q1 | SCREENED_NOT_PROMOTED | `02_代码/Q1_感官评分/configs/E004_gpu.yaml` | `03_结果/Q1_感官评分/q1_*_E004_GPU_screening.*`、`q1_screening_*_E004.*` | CUDA 累积有序 XGBoost 五项均未改善，不扩展至 10 次；仅作为有序路线的负结果保留 |
| E005 | Q1 | FULL_GATE_4_OF_5 | `02_代码/Q1_感官评分/configs/E005_continuous_ensemble.yaml` | `03_结果/Q1_感官评分/q1_*_E005.*`、`q1_full_*_E005.*` | 正式 10 次 OOF 为 R²/MAE/Spearman/代理 QWK/代理 ACC±1=0.2482/0.3397/0.4273/0.4150/0.5523；前四项过线且 10/10 胜出，ACC±1虽 9/10 胜出但未达 0.5734，故不单独晋级，作为 E006 连续路线 |
| E006 | Q1 | FULL_NOT_PROMOTED | `02_代码/Q1_感官评分/configs/E006_dual_route.yaml` | `03_结果/Q1_感官评分/q1_*_E006.*`、`q1_full_*_E006.*` | 正式 10 次 OOF 为 R²/MAE/Spearman/代理 QWK/代理 ACC±1=0.2480/0.3356/0.4314/0.4194/0.5552，五项均 10/10 次优于 E003；前四项通过绝对门槛，但 ACC±1 未达到 0.5734，故不晋级并按冻结顺序触发 E007 |
| E100 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E100_contract.yaml` | `03_结果/Q2_关键化合物/q2_candidate_dictionary.csv`、`q2_graph_*` | 候选、名称和有符号 Spearman 图合同通过；218 个严格化合物候选 |
| E101–E104 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E101_E104_selection.yaml` | `03_结果/Q2_关键化合物/q2_*_E101_E104.*` | 10×3 OOF 完成；E103 宏 MAE 0.3529，E104 0.3530；图惩罚未得到预测增益，扩展参数敏感性进入 E109 |
| E109 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E109_boundary_sensitivity.yaml` | `03_结果/Q2_关键化合物/q2_e109_*` | 扩展网格下 E103 的指标与 Top-3 完全不变；E104 未改善 MAE，图惩罚降为消融组 |
| E105 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E105_extra_trees_permutation.yaml` | `03_结果/Q2_关键化合物/q2_*_E105.*` | ExtraTrees 宏 $R^2$/MAE=0.2359/0.3422，10/10 次重复优于 E103；预测主线晋级，字段 Top-3 待共识与稳定性审计 |
| E106 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E106_rank_consensus.yaml` | `03_结果/Q2_关键化合物/q2_*_E106.*` | 生成 50×3 三方法统计共识；S1/S2/S3=41/90/19，PLS 敏感性 Jaccard=0.652，尚需 E107/E108 |
| E107 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E107_bootstrap_stability.yaml` | `03_结果/Q2_关键化合物/q2_bootstrap_*_E107.*`、`q2_stable_module_*_E107.csv` | 500 次风格内 bootstrap 完成；150 条候选入选频率中位数 0.428，≥0.6/0.7/0.8 为 43/31/23 条，两两 Jaccard 均值 0.245；需 E108 与化学证据分级 |
| E108 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E108_style_sensitivity.yaml` | `03_结果/Q2_关键化合物/q2_*_E108.*` | 匹配 10×3 原始/风格校正 Top-3 Jaccard=0.264，仅 2/50 完全一致；22 次留一风格删除 Jaccard=0.523，E106 候选平均保留率 0.641，确认风格混杂不可忽略 |
| E111 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E111_robust_consensus.yaml` | `03_结果/Q2_关键化合物/q2_*_E111.*` | 四路秩融合得到 150 条统计 Top-3，R1/R2/R3=38/52/60；目标预测 P1/P2/P3=22/12/16，进入独立化学证据分级 |
| E112 | Q2 | COMPLETE | `02_代码/Q2_关键化合物/configs/E112_chemical_evidence.yaml` | `03_结果/Q2_关键化合物/q2_*_E112.*` | 冻结排名后追加 OAV、描述词与文献核验；A/B/C/D=8/16/78/48，73 条可算 OAV，23 条中位 OAV≥1；不作因果表述 |

每次运行追加新行，不覆盖旧实验。指标必须链接到机器生成文件，论文数字以结果目录中的冻结版本为准。

