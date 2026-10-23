# Q2 关键化合物源码

本目录用于每个感官维度 Top-3 化合物、bootstrap 入选频率、方法间 Jaccard、共线化合物组和化学合理性证据。任何相关网络、特征筛选与重要性模型必须在重采样训练子集中重算。

当前状态：Q1 的 OOF 模型和公共预处理契约已冻结；路线 B 已获确认。E100—E105 及 E109 已完成。E105 ExtraTrees 的宏平均 $R^2$/MAE/Spearman 为 0.2359/0.3422/0.4144，在 10/10 次重复中均优于 E103；但 Top-3 入选频率中位数只有 0.40，需继续 E106 共识、E107 bootstrap 与 E108 风格校正。

运行 E100：

```bash
conda run --no-capture-output -n mathorcup \
  python 02_代码/Q2_关键化合物/src/run_e100_contract.py
```

运行 E101—E104 完整 OOF：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n mathorcup \
  python 02_代码/Q2_关键化合物/src/run_e101_e104.py \
  --repeats 10 --run-label ''
```

该流程全程 CPU；预处理顺序为训练折 1%–99% Winsor 截尾、中位数填补、缺失指示和标准化。

运行 E105 折外条件/模块置换：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n mathorcup \
  python 02_代码/Q2_关键化合物/src/run_e105_extra_trees_importance.py \
  --repeats 10
```

测试：

```bash
conda run --no-capture-output -n mathorcup \
  python -m unittest discover -s 02_代码/Q2_关键化合物/tests -v
```

