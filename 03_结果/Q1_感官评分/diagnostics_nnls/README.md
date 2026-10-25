# NNLS 准备诊断

这里只存放已有 E003/E005 OOF 的开发诊断，不是 E009 实验结果。

- `q1_complementarity_pairs.csv`：每重复、每目标的成对残差/秩相关与 ±1 错误互救，5 模型共 5,000 行；
- `q1_complementarity_summary.csv`：10 对模型的目标—重复宏平均；
- `q1_base_model_metrics_by_repeat.csv`：5 模型×10 重复，禁止与 mean-OOF 混报；
- `q1_base_model_metrics_by_target.csv`：5×10×50 的逐目标指标；
- `q1_grouped_base_metrics.csv`：冻结目标字典五感官组的宏平均；
- `q1_tail_target_baseline_definition.csv`：只由 E003 选出的最弱 13 个目标；
- `q1_tail_base_metrics.csv`：固定尾部目标上的各基模型均值；
- `q1_complementarity_manifest.json`：原始 OOF/目标字典/脚本哈希与 `diagnostic_only=true`。

`first_rescued_by_second` 是第一模型失败样本中被第二模型救回的比例，先逐目标、
逐重复计算，再宏平均。它不是融合模型的精度。`oracle_union_pm1_diagnostic` 假设
预知每个样本该信谁，只是无法部署的诊断上限，严禁写成候选成绩。

重算命令（不拟合模型）：

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/mary/miniconda3/envs/mathorcup/bin/python \
  '02_代码/Q1_感官评分/src/stacking_diagnostics.py'
```

E009 主候选预注册为 `nnls_group`；shared/target 是对照。E007 完整审计前，
正式 E009 运行入口应拒绝执行，不需要也不允许从这些诊断 CSV 拟合元权重并自评。
