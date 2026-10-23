# Q2 关键化合物源码

本目录用于每个感官维度 Top-3 化合物、bootstrap 入选频率、方法间 Jaccard、共线化合物组和化学合理性证据。任何相关网络、特征筛选与重要性模型必须在重采样训练子集中重算。

当前状态：Q1 的 OOF 模型和公共预处理契约已冻结；路线 B 已获确认。E100 数据、候选与图合同已完成并测试通过，下一步执行 E101—E104。当前尚未启动 500 次 bootstrap。

运行 E100：

```bash
conda run --no-capture-output -n mathorcup \
  python 02_代码/Q2_关键化合物/src/run_e100_contract.py
```

测试：

```bash
conda run --no-capture-output -n mathorcup \
  python -m unittest discover -s 02_代码/Q2_关键化合物/tests -v
```

