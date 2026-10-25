# Q3 风格生成源码

本目录用于以 `tasting_category_fine` 为条件生成 Tripel、Kriek、Lambic 的化学/感官画像，并求解跨风格迁移。DMS、Diacetyl、4VG、trans-2-nonenal 和 stale hops 作为显式缺陷约束，不允许因迁移而恶化。

当前状态：数据契约、经验基线、收缩高斯/Copula、共享低秩消融和受约束迁移均已完成。共享低秩在相关结构误差上略有改善，但边缘 W1、联合 energy 和覆盖率未超过收缩 Copula，因此 Copula 作为迁移统计基线。迁移已对 11 款源啤酒、3 个目标风格和 C1/C2/C3 分别登记目标靠近、改动、逐项约束、Pareto 与拒绝；Stale hops 代理和最终感官画像仍待 E205，当前不冻结最终候选。Q1/Q2 冻结结果不受本工作树影响。

2026-10-25准备检查点：E200数据契约、C1/C2/C3缺陷审计和无训练分布评价代码已实现。只读取官方附件做契约，未拟合模型、未使用GPU。

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/mary/miniconda3/envs/mathorcup/bin/python -m unittest discover \
  -s 02_代码/Q3_风格生成/tests -q
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/mary/miniconda3/envs/mathorcup/bin/python \
  02_代码/Q3_风格生成/src/run_e200_contract.py --write
```

E200输出8个契约文件到`03_结果/Q3_风格生成/`。`q3_field_contract_E200.csv`识别37/33/161个字段；`q3_calibration_conflicts_E200.csv`记录80个正值不一致和3个负参考单元。该文件含原始审计值，不是清洗后建模表。源缺陷44记录中33项可比、9项零/地板歧义、2项缺失，不把此比例说成迁移成功率。

模块：`q3_contract.py`（键/单位/测量状态），`q3_constraints.py`（显式约束、投影与拒绝），`q3_evaluation.py`（W1/energy/相关/覆盖/近邻重复诊断）。后者只在合成测试中运行，评价尺度由调用者训练折传入；不包含C2ST、生成器、迁移求解器或最优模型结论。

主候选为共享低维结构+风格收缩+受约束迁移，经验生成、Copula/全维收缩和小型条件深度模型对照。完整定义、拟定E200—E207及验证边界见 `docs/analysis/2026初赛/Q3_独立方案与优化路线.md`。

E203 通过 `run_e203_shared_lowrank.py` 运行，秩和残差收缩强度只在外层训练折内选择；`audit_e203.py` 使用训练折 1%—99% 分位作稳健敏感性，并按风格×重复与 Copula 成对比较。普通缺失不冒充已知源浓度；Stale hops 只能先作感官预测风险检查；C1 严格相对不升、C2 超阈负担不升、C3 不升且阈下分别登记。不得把零迁移算成功，也不得仅统计成功候选后宣称全样本安全。

E204 通过 `run_e204_constrained_migration.py` 执行全数据后选择设计；`audit_e204.py` 检查候选数量、哈希、已知直接约束、零迁移、未知状态和代理门控。其全数据结果不是新外层验证；模型选择仍以 E202/E203 折外结果为依据。

