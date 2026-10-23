# 发现与决策：A题问题二

## 需求
- 详细分析问题二，比较多条可能最优路线。
- 参考本地全部相关文档、历年优秀论文、学术论文、GitHub、Kaggle、CSDN。
- 延续 Q1 的数据与验证口径；当前不启动训练。

## 本地研究发现
- 题目要求用 `2.csv` 构建化合物共表达图，结合 `1.csv` 给 50 个感官维度各自的 Top-3 关键化合物，并用 bootstrap 入选频率或方法间 Top-3 Jaccard 量化稳定性，再作化学合理性论证。
- `1.csv` 为 250 行、234 列：`beer`、`beer_id`、`tasting_category_fine` 加 231 个化学/理化字段。Q2 延续 Q1：用 `beer` 连接，`beer_id` 不连接也不建模。
- `2.csv` 是 231×231 的对称相关矩阵，无缺失、对角线为 1。它与从 `1.csv` 逐列重算的 Spearman 矩阵逐元素一致，最大绝对差 `6.66e-16`；与 Pearson 的最大绝对差为 `1.1234`。题面叙述中的 Pearson 与附件说明冲突，主文应按经数据验证的 Spearman 口径书写。
- `4.csv` 提供 250×50 个标准化感官目标；`5.csv` 是 50×50 的感官 Spearman 矩阵；`6.csv` 实际为 44×50，而非题面所述 46×50，并且等于对应化学字段与感官字段在全样本上的 Spearman 相关。`6.csv` 只能用于模型完成后的后验核验，不能参与 Top-3 筛选，否则会泄漏全样本标签。
- `01_数据/compound_name_map.csv` 已覆盖 231 个字段且一一映射，可作为英文名、字段代码、化学名称和单位的唯一连接表。
- `8.xlsx` 含感官阈值、化学名称、off-flavor/aging 标记、预计算 GBM MDI/SHAP、73 个校准化学字段和价格等。阈值与化学注释可用于合理性论证；预计算 MDI/SHAP 的目标、划分和训练过程未说明，不能作为主排名或性能证据。
- 231 个预测量并非全是单体化合物，还包含 pH、色度、苦度、乙醇、CO2 等理化指标，以及 acids/esters/hop aroma/sulfur 等聚合量。题面要求“Top-3 化合物”，必须先建立候选资格表。建议主表只纳入可识别的单体化合物；理化指标与聚合量另作补充因素表或机制锚点，避免用 pH、苦度等代替化合物作答。
- `2.csv` 的绝对相关阈值非常敏感：`|rho|>=0.2/0.3/0.4/0.5/0.6` 分别产生 8323/3918/1660/645/266 条边；到 0.5 已有 48 个孤立点，且 645 条边中仅 16 条为负边。单一硬阈值会改变连通性并抹去大部分负相关结构，应采用训练折内重构、保留符号的自适应稀疏化，并做阈值/邻居数敏感性分析。
- Q1 已冻结交叉验证划分和预处理口径。Q2 的预测性比较应复用相同 OOF 划分；任何监督筛选、图构建、相关簇划分和重要性计算都要在训练折内完成。
- Q1 文档已明确：相关、预测贡献和因果效应是三个不同口径。Q2 最终统一称“稳定关键预测关联化合物”或“候选关键化合物”，不在无干预数据下声称因果。
- 历年优秀论文总结显示，高分答卷强调每问独立的“难点—对策”、多模型对比、交叉验证、结果数字、消融/敏感性和业务/机理解释。对 Q2 的迁移重点不是堆模型，而是把共线性、全量矩阵泄漏、Top-3 离散不稳定和化学证据链写清楚。

## 外部研究发现
- 原始数据论文为 Schreurs et al., *Predicting and improving complex beer flavor through machine learning*, Nature Communications 15, 2368 (2024), DOI `10.1038/s41467-024-46346-0`。其样本规模、22 个风格、250 款啤酒、226 个化学性质和 50 个训练品评维度与赛题附件一致，因此它是本题最重要的一级来源。
- 原论文比较 10 类模型，树模型总体优于线性模型和人工神经网络；论文明确将 ANN 的中等表现归因于数据量较小。这个结果支持 Q2 以稀疏/树模型为主、GNN 只作后续消融。
- 原论文用 GBR 的 MDI 与 SHAP 拆解模型，并做了 100 次 GBR/RF/ET 重复；作者也明确指出共相关变量会使单个变量的 MDI/SHAP 重要性被压低，ExtraTrees 的 Top predictors 变化更大。这与赛题新增 bootstrap/Jaccard 要求高度吻合，也说明不能直接采用附件 8 的一次性 MDI/SHAP。
- 原论文的化学加标实验只对少数候选化合物形成了较强因果验证；本题无法为 50×3 个候选复现加标实验，因此除原论文已验证的个别化合物外，其余一律表述为预测关联和化学合理性，而非因果驱动。
- Zenodo 官方仓库 `10.5281/zenodo.10653704` 提供训练代码、SHAP notebook 和全部补充表。代码显示 10 类模型及 70/30 风格分层划分；同时 `generate_X(impute=True)` 在 `train_test_split` 之前对全数据均值填补。赛题实现应沿用模型列表作基线参考，但把填补、缩放、构图和筛选全部移入训练折。
- 附件 8 的 `GBM importance MDI` 与 `GBM SHAP Values` 正是原仓库的 RateBeer overall appreciation 解释产物：SHAP 表含 250 行 SHAP + 250 行 concentration，且只有一个 `base_prediction`。它不是 50 个训练品评维度的逐目标结果，不能作为 Q2 答案。
- Langfelder & Horvath (2008) 的 WGCNA 给出 `a_ij=|cor_ij|^β` 的软阈值加权网络、模块和模块特征向量。对本题可借用“软权重 + 模块”的思想，但不应机械套用转录组的无标度拓扑阈值；β 应按本题的重采样网络稳定性和下游预测表现选择。
- Li & Li (2008) 的网络约束回归（Grace）将 L1 稀疏惩罚与图拉普拉斯平滑结合，适合 p≈n 且特征有已知图结构的场景。自适应 Grace 可让负相关邻居的系数倾向相反方向，适合本题有符号 Spearman 图。
- Meinshausen & Bühlmann (2010) 的 stability selection 说明重采样与变量选择可结合以提升高维结构估计的可复现性。题面指定普通 bootstrap 时，入选频率可作为主要量化；若要引用其错误控制理论，必须另做半样本 stability selection，不能把普通 bootstrap 频率直接宣称为有限样本错误率控制。
- Strobl et al. (2008) 证明随机森林的常规变量重要性受相关预测量影响；scikit-learn 官方文档也指出强相关变量会互相替代，使单变量 permutation importance 同时偏低。故本题采用“相关模块联合置换 + 模块内稀疏线性代表”的双层归因。
- Aas et al. (2021) 指出在相关特征下独立性假设会使 SHAP 解释失真；Apley & Zhu (2020) 的 ALE 避免 PDP 在相关预测量下的外推问题。因此 SHAP 只作辅助，效应方向优先用交叉拟合 ALE/条件效应及稀疏模型系数。
- Robust Rank Aggregation 可整合多个有噪声的排序；本题可使用中位秩或 RRA 形成共识榜，但最终还要保留各方法原始排名与 Jaccard，不能用聚合排名掩盖分歧。
- Kaggle `top-beer-information` 本地副本为 5558×21、112 种风格，11 个感官字段是评论词频，没有化学浓度，也无法通过名称或条码与官方 250 款啤酒连接。它可用于 Q3 或风格级感官合理性，不能验证 Q2 的具体 Top-3 化合物。
- GitHub/官方开源实现方面，已核对 scikit-learn 的 permutation importance 文档和 CRAN WGCNA 镜像；本地 Python 已有 sklearn、ExtraTrees、SHAP、NetworkX、SciPy，可在不新增依赖的情况下实现主线。R 环境没有 WGCNA/igraph/glmnet；R 仍只负责最终 ggplot2 图形。
- CSDN 的 WGCNA/SHAP 实战文章可用于了解常见操作顺序，但多采用固定 `R²>0.8/0.9` 的无标度阈值或直接读取单次 SHAP 排名。该做法缺少针对本题化学网络的验证，因此不进入证据链。

## 技术决策
| 决策 | 理由 |
|---|---|
| 主任务拆成“候选界定—图构建—预测归因—稳定选择—化学核验”五层 | 防止把边际相关、模型重要性和化学机制混为一谈 |
| 正式模型评价时折内重构 Spearman 图 | `2.csv` 使用全样本，直接用于交叉验证内的监督流程会让验证样本参与结构估计 |
| 同时报 exact compound stability 与 community stability | 共线化合物会在重采样中互相替代，仅按字段统计会低估机制模块的稳定性 |
| `6.csv` 与 `8.xlsx` 预计算结果仅作后验一致性检查 | 它们由全样本或未知训练过程产生，不能作为候选筛选器 |
| 深度模型只作为后续备选消融 | 250 个样本不足以支撑 GNN 作为主线；Q2 的首要评分点是稳定、可解释的 Top-3 |

## 风险与问题
| 问题 | 处理方向 |
|---|---|
| 强共线导致单变量重要性不稳定 | 比较图约束稀疏模型、组条件置换和多方法共识 |
| 相关不等于因果 | 统一使用“关键预测关联化合物”，化学合理性作独立证据 |
| 2.csv 为全数据相关矩阵 | 预测评价阶段折内重构图；最终解释阶段再使用官方全图 |
| Top-3 的 Jaccard 取值过粗 | Top-3 Jaccard 按题面保留，同时报告 Top-10 秩相关、精确入选频率和社区级一致性 |
| 化合物名称、单位和阈值不完全齐全 | 先生成候选资格与证据覆盖表；缺阈值者不计算 OAV，只用可核验的类别/文献证据 |

## 资源
- 本地 mcup skill、BZD 建模思路 skill、赛事文档与 40 篇优秀论文文本。
- Nature Communications 原始论文：https://www.nature.com/articles/s41467-024-46346-0
- Zenodo 官方数据与代码：https://zenodo.org/records/10653704
- Stability selection：https://doi.org/10.1111/j.1467-9868.2010.00740.x
- WGCNA：https://doi.org/10.1186/1471-2105-9-559
- Network-constrained regularization：https://doi.org/10.1093/bioinformatics/btn081
- Conditional variable importance：https://pmc.ncbi.nlm.nih.gov/articles/PMC2491635/
- Dependent-feature SHAP：https://doi.org/10.1016/j.artint.2021.103502
- ALE：https://doi.org/10.1111/rssb.12377
- scikit-learn permutation importance：https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/permutation_importance.rst

## 视觉/浏览器发现
- 原始论文 Fig. 4 的 MDI/SHAP 解释对象是 RateBeer appreciation，并非本题 50 个训练品评目标；附件 8 的表结构与其官方 SHAP notebook 完全对应。
- 全局 231 节点网络不适合作为一张密集“毛线团”主图。正文更适合展示模块总览以及 4–6 个代表目标的 Top-3 邻域；50×3 全量结果用横向分组点图/热图与附表承载。
