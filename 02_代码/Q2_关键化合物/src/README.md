# Q2 src

已实现：

- `q2_contract.py`：附件读取、严格候选资格、折内 Spearman、软权边、有符号归一化图拉普拉斯与阈值审计；
- `run_e100_contract.py`：生成 E100 候选字典、图节点/边、数据合同和 manifest。
- `run_e101_e104.py`：折内 Spearman、PLS/VIP、ElasticNet 与图惩罚模型；
- `run_e105_extra_trees_importance.py`：折外条件字段置换与模块联合置换；
- `run_e106_consensus.py`：三方法支持优先的倒数秩融合；
- `run_e107_bootstrap.py`：500 次风格内 bootstrap、OOB 重要性与稳定模块；
- `style_adjustment.py`、`run_e108_style_sensitivity.py`：无泄漏风格残差化与逐风格删除；
- `run_e108_raw_calibration.py`：相同 10×3 口径的未校正对照。

后续模块：总关联、bootstrap、风格校正与留一风格四路稳健秩融合，以及化学证据分级。

