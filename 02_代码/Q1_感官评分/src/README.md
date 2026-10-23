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

