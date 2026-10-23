# Q2 关键化合物源码

本目录用于每个感官维度 Top-3 化合物、bootstrap 入选频率、方法间 Jaccard、共线化合物组和化学合理性证据。任何相关网络、特征筛选与重要性模型必须在重采样训练子集中重算。

当前状态：Q1 的 OOF 模型和公共预处理契约已冻结；路线 B 已获确认。E100—E109 的主线与敏感性实验已完成。匹配 10×3 原始/风格校正 Top-3 Jaccard 为 0.264，说明最终证据必须区分总关联与风格内关联；下一步为四路稳健秩融合和化学证据分级。

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

运行 E108 风格校正与逐风格删除，并补匹配原始数据对照：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n mathorcup \
  python 02_代码/Q2_关键化合物/src/run_e108_style_sensitivity.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n mathorcup \
  python 02_代码/Q2_关键化合物/src/run_e108_raw_calibration.py
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

