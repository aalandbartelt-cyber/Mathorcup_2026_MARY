# Q2 configs

配置记录重采样次数、候选方法、软权图参数、Top-k、稳定频率阈值、FDR 阈值和随机种子。`E100_contract.yaml` 已冻结数据、候选资格、Spearman 图和训练折隔离口径。

`E113_partial_graph.yaml` 冻结稀疏偏相关图候选：每个外层训练折内先插补并作
秩高斯变换，再用 3 折内层验证选择 Graphical Lasso 的共享 `alpha`。选择顺序为
对数似然 1-SE 可行集、边 Jaccard、较大 `alpha`；Louvain 只读取该训练折的偏
相关图。完整数据图只能标为 `full_data_descriptive`，不得参与外层验证或排名。

