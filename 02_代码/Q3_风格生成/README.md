# Q3 风格生成源码

本目录用于以 `tasting_category_fine` 为条件生成 Tripel、Kriek、Lambic 的化学/感官画像，并求解跨风格迁移。DMS、Diacetyl、4VG、trans-2-nonenal 和 stale hops 作为显式缺陷约束，不允许因迁移而恶化。

当前状态（2026-10-26）：E200—E205B/E207—E211均已完成并有审计，E206深度挑战未运行。共享低秩、经验Beta/整行混合联合选择、秩空间相关选择、化学空间PCA/RBF局部混合均未满足综合替换条件，保留风格收缩Copula。E211已将旧混合的共享随机流修订为选择/Copula/局部独立流；旧成绩只读保留，新普通混合energy降幅0.004149未达既定0.005，基函数无增量，不启动神经门控。最终NNLS画像与迁移不变：原主规则2/33（Tripel/Kriek，均源230），Lambic主规则0/11、最大代理裕量辅助规则1条，不能合并分母。未知LOD/源缺失仍未知，不能称物理安全。Q1/Q2冻结结果不受本工作树影响。

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

模块：`q3_contract.py`（键/单位/测量状态），`q3_constraints.py`（显式约束、投影与拒绝），`q3_evaluation.py`（W1/energy/相关/覆盖/近邻重复诊断）。评价原语已在真实外折和合成测试运行，尺度由当前训练折传入；生成器、迁移与替换判断各有独立runner/audit，C2ST尚未运行。

原共享低维设计现为未采用的消融，当前主线为经验边缘+风格收缩Copula+受约束搜索，小型条件深度模型未运行。完整定义、拟定E200—E207及验证边界见 `docs/analysis/2026初赛/Q3_独立方案与优化路线.md`。

E203 通过 `run_e203_shared_lowrank.py` 运行，秩和残差收缩强度只在外层训练折内选择；`audit_e203.py` 使用训练折 1%—99% 分位作稳健敏感性，并按风格×重复与 Copula 成对比较。普通缺失不冒充已知源浓度；Stale hops 只能先作感官预测风险检查；C1 严格相对不升、C2 超阈负担不升、C3 不升且阈下分别登记。不得把零迁移算成功，也不得仅统计成功候选后宣称全样本安全。

E204 通过 `run_e204_constrained_migration.py` 执行全数据后选择设计；`audit_e204.py` 检查候选数量、哈希、已知直接约束、零迁移、未知状态和代理门控。其全数据结果不是新外层验证；模型选择仍以 E202/E203 折外结果为依据。

