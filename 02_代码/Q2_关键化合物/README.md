# Q2 关键化合物源码

本目录用于每个感官维度 Top-3 化合物、bootstrap 入选频率、方法间 Jaccard、共线化合物组和化学合理性证据。任何相关网络、特征筛选与重要性模型必须在重采样训练子集中重算。

当前状态：Q1 的 OOF 模型和公共预处理契约已冻结；路线 B 已获确认。E100—E107 及 E109 已完成。E107 的 500 次风格内 bootstrap 中，150 条候选的入选频率中位数为 0.428；尚需 E108 风格校正与化学证据分级。

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

运行 E107 正式 bootstrap：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n mathorcup \
  python 02_代码/Q2_关键化合物/src/run_e107_bootstrap.py \
  --n-bootstraps 500 --run-label ''
```

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

