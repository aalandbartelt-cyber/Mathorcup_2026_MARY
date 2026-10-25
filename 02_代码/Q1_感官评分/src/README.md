# Q1 src 计划

按下列边界实现，文件名在首轮编码时冻结：

1. 数据加载与 250×231、250×50、主键、列序断言；
2. 10×3 重复风格分层外层划分与内层调参划分；
3. E0—E3 基线；
4. E4—E7 低秩、树模型和目标自适应组合；
5. E8—E11 条件候选，满足晋级标准后才运行；
6. OOF 指标、共形区间、0—6 相对等级和新样本推理。

当前状态：数据契约、固定 CV、E001/E002/E003 CPU 基线已实现。数据契约同时生成
`01_数据/compound_name_map.csv`，将附件 2 的 231 个英文显示名映射到附件 1
字段代码，并以附件 8 的 `field_name_R` 做完整性复核。运行：

```bash
conda run -n mathorcup python 02_代码/Q1_感官评分/src/data_contract.py
conda run -n mathorcup python 02_代码/Q1_感官评分/src/run_e001_baselines.py
conda run -n mathorcup python 02_代码/Q1_感官评分/src/run_e002_robust_linear.py
conda run -n mathorcup python 02_代码/Q1_感官评分/src/run_e003_extra_trees.py
conda run -n mathorcup python 02_代码/Q1_感官评分/src/compare_q1_models.py
conda run -n mathorcup python 02_代码/Q1_感官评分/src/prepare_q1_figure_data.py
/home/mary/miniconda3/envs/r/bin/Rscript 02_代码/Q1_感官评分/src/plot_q1_drafts.R
conda run -n mathorcup python -m unittest discover -s 02_代码/Q1_感官评分/tests -v
```

第二轮候选及审计入口：

- `run_e004_ordinal_hgb.py`：累计有序 HGB（含筛查与检查点）；
- `run_e004_gpu.py`：CUDA 累计有序 XGBoost 消融；
- `run_e005_continuous_ensemble.py`：ExtraTrees/HGB/Ridge/PLS2 折内连续融合；
- `run_e006_dual_route.py`：连续路线与累计有序路线折内融合；
- `run_e007_shrunk_gating.py`：逐目标权重向全局权重收缩的门控；
- `audit_q1_second_round.py`：筛查、正式五门槛及最终 bootstrap/尾部/风格审计。

所有长任务按一个外层折原子更新 `q1_checkpoint_*.npz`，恢复时会校验配置、数据
和 CV 签名。只有 manifest 的 `status=complete` 才能进入正式审计，部分检查点不
得解释为实验结果。

