# Q2 configs

配置记录重采样次数、候选方法、软权图参数、Top-k、稳定频率阈值、FDR 阈值和随机种子。`E100_contract.yaml` 已冻结数据、候选资格、Spearman 图和训练折隔离口径。

`E113_partial_graph.yaml` 冻结稀疏偏相关图候选：每个外层训练折内先插补并作
秩高斯变换，再用 3 折内层验证选择 Graphical Lasso 的共享 `alpha`。选择顺序为
对数似然 1-SE 可行集、边 Jaccard、较大 `alpha`；Louvain 只读取该训练折的偏
相关图。完整数据图只能标为 `full_data_descriptive`，不得参与外层验证或排名。

`E116_grouped_multitask.yaml` 仅登记代码就绪状态，尚未跑正式数据。三个互不重叠的
malt/hops/esters 组各包含对应 Aroma/Flavour 目标，其余 20 目标使用独立
ElasticNet；内层每次重新拟合中位数/缺失指示、X/Y 缩放和小网格参数。
零系数即使在完整排名中占位，也不会计为 Top-3 入选。运行入口要求显式
`--execute` 且 E113/E114/E115 已完成；本轮用户只授权代码和测试，不执行该选项。
正式 OOF 完成后仍需稳定性/四个直接锚点审计，预测指标单独通过不能晋级。

