# Q3 E211 化学基函数邻域混合实施计划

> 执行方式：executing-plans，本会话逐项实现与核验，沿用q3-foundation隔离工作树。用户已确认候选设计并要求开始，无额外确认点。

**Goal:** 以原始局部混合为强基线，检验化学低维坐标与多尺度基函数是否提供生成增量。

**Architecture:** 输出仍为231字段的原尺度谱；几何只用218严格单体，训练折1%—99%剪裁用于几何、IQR标准化、4维PCA、8个KMeans原型及0.5/1.0倍原型间正距离中位数的两尺度Gaussian RBF。原生成/评价/缺陷约束文件均不修改，只新增模块与结果。

**Tech Stack:** numpy/scipy/pandas/sklearn；CPU单线程；Q1只读签名，无新NNLS拟合、无Q2训练、无GPU神经模型。

## 1. 冻结模型与选择

A：冻结Copula生成谱及评价读取E202/E208。

**运行前代码审查修订（无新实验结果）：** 原E208混合中选择器与两分量分别重新初始化相同seed，不保证指示变量与分量独立。一个U和2+U的二分量反例中，共用U选择会使P(X<0.25)=0，而独立混合应为0.125。旧结果作为实际算法成绩保留，不将其混合密度解释当已验证。新B/C/D统一使用`SeedSequence(seed).spawn(3)`派生选择/Copula/局部三个独立随机流，跨α与几何沿用同一组流作为共同随机数；旧`mixture_selected`谱/指标增加为只读`legacy_raw_mixture`，不重跑/覆盖。

B `raw_mixture`：原231字段/训练std距离，3近邻、原插值系数Uniform(0.2,0.8)，整行混合局部概率{0.25,0.5,0.75}。局部采样原语须与旧实现同seed等价，完整混合采用上述独立流；与旧E208的混合差异单列，不能借重现旧9/10成绩作验收。

C `pca_mixture`：只把局部邻居改为上述218单体的4维坐标距离；原插值、同风格/非自身、邻居数和整行混合保持。

D `basis_mixture`：只把C的邻域排序改为两尺度RBF距离。支持先限定在PCA最近min(6,n_s−1)个邻居，RBF再取最近min(3,n_s−1)个；基函数全部近零或距离退化则回退PCA。对同风格训练邻域，用训练重采样5次、维数3/5和std替代IQR共8项扰动算逐锚点Jaccard；若中位数<0.5，D在该风格整体回退C并登记，不能说基函数全程有效。

PCA、原型、带宽、稳定性、插补全在当前内/外训练范围内重估。稳定性在原训练点上比较邻域，不访问留出。输出谱不先做缺陷投影或Q1筛选后再评价。

每个B/C/D单独在内层2折、3种子选择相同α网格；损失沿用稳健W1+energy+0.25×绝对对数离散误差，原始复制率>0或坍缩排除。同分选较少局部分量。不按外折拼选家族/α。

## 2. 预登记决定

沿用10×3、3风格、5种子和既有主模型条件：raw W1增幅≤0.01、energy≤0.02；robust W1增幅≤0.01、energy至少降0.005且至少7/10重复改善、coverage至少增0.05、log-spread-error增幅≤0.05；每风格robust W1/energy增幅≤0.02；复制与坍缩为0。

**确定的回退顺序，不运行后改写：**

1. D满足上述条件，且对B、C均满足以下增量条件，才采用D并声称基函数有增量：robust energy至少降0.002、至少7/10重复改善，W1增幅≤0.005、coverage下降≤0.02、log-spread增幅≤0.02，每风格W1/energy增幅≤0.02；每风格至少50%的外折实际启用RBF（非回退）。
2. 否则，若新独立随机流B满足原主模型条件，选择简单B作为**探索性最终候选**；E208旧决定保持不变，明确旧共享随机流混合性能先前已被观察，不能当新B已有成绩或称本轮独立确认。
3. 否则保持原Copula。C只作几何消融，本轮不事后改选C。D仅超过A却未超过B/C，不足以证明基函数必要。

不把外折复用称未触碰的独立测试，不据此作确认性显著性推断。新分布候选不自动替换既有900份NNLS画像/迁移结果，后续全数据拟合与约束验证另行登记。

## 3. 任务与文件

- [ ] 新增`02_代码/Q3_风格生成/tests/test_basis_mixture.py`，先运行显示缺少模块/API，再实现`src/basis_mixture.py`。核验几何只见训练、尺度与常量、原型/种子可复现、同风格非自身邻居、远点回退、完整231字段、整行混合、B等价性、谱内插来源。
- [ ] 新增`tests/test_e211_runner.py`并验证失败，再实现`src/run_e211_basis_mixture.py`。config严格校验、内层选择、5种子匹配数、30折恢复、训练beer、哈希、邻域/稳定性表、逐字段W1。已完整运行时只验哈希并返回，不覆盖正式结果。
- [ ] 新增`tests/test_e211_audit.py`，再实现`src/audit_e211.py`。覆盖四组+只读旧混合/30折/完整选择表、旧谱指标复现、均值和重复/风格分解、同规则替换判断、基函数增量、预登记回退、manifest/候选注册。
- [ ] 运行首折并审计后提交，完整30折后正式审计。使用`env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/mary/miniconda3/envs/mathorcup/bin/python -m pytest '02_代码/Q3_风格生成/tests' -q`，所有测试通过后提交源码。
- [ ] 报告不利指标、稳定性回退与来源，不声称真实安全；更新对照总表和路线，在交接/决策时写to_zcode。每段commit/push，不提交大型缓存。

## 4. 实现接口与核验示例

`ChemicalBasisGeometry.fit(x, geometry_indices, coordinate_dim=4, prototype_count=8, seed=20261026)`只接受当前训练化学矩阵；`transform(x)`为out-of-sample坐标，`basis(z)`输出多尺度权重。`ChemicalMixtureGenerator.fit(x, styles, geometry_indices, config)`拟合三种同风格邻域；`local(style,n,geometry,seed)`保存锚点与t；`generate(style,n,geometry,alpha,seed)`整行抽分量。原Copula仍复用既有类，不改其源码。

核心测试应检查：

```python
raw = model.local('Tripel', 40, 'raw', seed=11)
old = original.generate('Tripel', 40, method='local_interpolation', seed=11)
np.testing.assert_array_equal(raw['values'], old['values'])
mixed = model.generate('Tripel', 40, 'raw', alpha=0.5, seed=11)
assert np.all(np.all(mixed == raw['values'], axis=1) | np.all(mixed == copula, axis=1))
```

审计结果必须能由冻结CSV重算；原B重现容差10^-9用于W1/energy/coverage/复制，秩相关保留已登记的CSV浮点敏感性，不能以该例外放松主指标。
