# Q1 感官评分源码

目标：以 `1.csv` 的 231 个化学/理化字段预测 `4.csv` 的 50 个聚合感官 z 分数，并输出 0—6 相对强度等级、不确定区间和逐目标折外指标。

目录约定：

- `src/`：数据契约、CV、基线、增强模型、评价和推理源码；
- `configs/`：实验 YAML，不在代码中硬编码模型参数；
- 结果写入 `03_结果/Q1_感官评分/`；
- R 绘图数据写入结果目录，成图写入 `04_图表/Q1_感官评分/`；
- 论文可直接使用的文字、表格说明与图表清单写入 `05_论文/Q1_感官评分/`。

输入主键只用 `beer`。所有填补、缩放、特征筛选、投影、混合权重、区间和量表映射必须在训练折内拟合。

当前只读/预处理命令：

```bash
conda run -n mathorcup python 02_代码/Q1_感官评分/src/data_contract.py
conda run -n mathorcup python 02_代码/Q1_感官评分/src/make_cv_splits.py
```

两条命令不训练模型，分别核验数据契约和生成冻结的外层划分。

第二轮多指标候选遵循“3 次筛查重复→通过后 10 次正式重复”的顺序，禁止用
筛查结果代替正式结论：

```bash
conda run -n mathorcup python 02_代码/Q1_感官评分/src/run_e004_ordinal_hgb.py --stage screening
conda run -n mathorcup python 02_代码/Q1_感官评分/src/run_e005_continuous_ensemble.py --stage full
conda run -n mathorcup python 02_代码/Q1_感官评分/src/run_e006_dual_route.py --stage full
conda run -n mathorcup python 02_代码/Q1_感官评分/src/run_e007_shrunk_gating.py --stage full
conda run -n mathorcup python 02_代码/Q1_感官评分/src/audit_q1_second_round.py --mode final
```

正式晋级必须同时满足五项绝对门槛和五项 `7/10` 重复非劣条件。最终审计另做
20,000 次配对啤酒 bootstrap、E003 预先定义的尾部目标审计和逐风格误差审计；
结果写入 `03_结果/Q1_感官评分/q1_second_round_*`，不会覆盖 E001—E003。

