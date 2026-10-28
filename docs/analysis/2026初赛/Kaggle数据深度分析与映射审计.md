# Kaggle 数据深度分析与映射审计

> 日期：2026-10-25  
> 性质：**原始数据一致性审计**。只比较官方 `4.csv` 风格均值与 Kaggle 风格画像，不使用 Q1/Q2/Q3 模型预测，不训练、不调参、不改变任何主结果。  
> 目的：判断这份 Kaggle 数据值得在哪些维度、哪些风格上作为外部校验，避免 Q2/Q3 收口后盲目使用。

## 1. 数据与对象

- 官方：250 款啤酒、22 个 `tasting_category_fine`、50 个标准化感官目标；本文只用 9 个可代理维度和 ABV/Overall 做数据级对照。
- Kaggle：5,558 行、112 风格；11 个评论词频维度；132 行全零描述符已排除出风格画像。
- 官方风格样本量差异很大：Faro 3、Brut 4、Flanders old brown 5、Scotch 5、Brett/cofermented 6、West Flanders ale 7、Low/No alcohol 8、Christmas 9、Kriek 10、Lambic 11、Pils/Lager 11、Amber/Brown/Wheat 12、Stout/Porter 13、Hoppy 17、Strong ale 18、Tripel 28、Blond 31。
- Kaggle 可用行数：Faro 14/16、Kriek 19、Brut 36/50、Lambic Traditional 40/49；其余多数 49–50 或合并池 100–700。

## 2. 方法

1. 官方 `4.csv` 按风格求 9 个代理目标均值：`body`、`alcohol`、`bitternes`、`sweetness`、`acidity`、`F_hops_all`、`F_malt_all`、`F_esters_fruity`、`clove+coriander` 平均。
2. Kaggle 使用 `kaggle_official_style_profiles_draft.csv` 的 `mean_prop_*`；风格映射为 `primary_pool=1`。
3. 对每个维度，在 22 个官方风格上计算 Spearman 秩相关；5,000 次置换检验；留一风格（LOO）范围；并报告 `n_official>=5` 的敏感性结果。
4. 数值对照：官方 `10**ethanol` 风格中位数 vs Kaggle ABV 中位数；官方 `overall` 风格均值 vs Kaggle 平均评分。
5. 风味结构：5,426 行可用样本的 11 维 Spearman 相关与 PCA（4 个主成分）。

## 3. 结果

| 维度 | rho（22 风格） | 置换 p | rho（n≥5） | p（n≥5） | LOO 范围 | 角色 |
|---|---:|---:|---:|---:|---|---|
| Sour | **0.889** | 0.0002 | **0.878** | 0.0002 | 0.873–0.909 | 主外部校验 |
| Sweet | **0.605** | 0.0042 | **0.638** | 0.0030 | 0.545–0.687 | 主外部校验 |
| Bitter | **0.556** | 0.0072 | **0.534** | 0.0166 | 0.490–0.678 | 主外部校验 |
| Malty | **0.519** | 0.0154 | **0.571** | 0.0096 | 0.452–0.591 | 主外部校验 |
| Hoppy | **0.510** | 0.0184 | **0.462** | 0.0440 | 0.436–0.623 | 主外部校验 |
| Alcohol | 0.490 | 0.0206 | 0.418 | 0.0688 | 0.445–0.691 | 次级 |
| Spices | 0.453 | 0.0360 | 0.411 | 0.0692 | 0.374–0.557 | 次级 |
| Fruits | 0.447 | 0.0440 | 0.438 | 0.0506 | 0.364–0.519 | 次级 |
| Body | 0.268 | 0.2340 | 0.370 | 0.1036 | 0.174–0.392 | 仅探索 |

数值对照：

- **ABV**：rho = **0.775**，p = 0.0002，LOO 0.741–0.836。
- **Overall rating**：rho = **0.494**，p = 0.0220，LOO 0.418–0.599。

解释：

- **主外部维度**：Sour、Sweet、Bitter、Malty、Hoppy 方向为正、置换显著、留一风格后仍为正，适合作为 Q1/Q3 收口后的主外部校验。
- **次级维度**：Alcohol、Fruits、Spices 在 22 风格上 p<0.05，但剔除 2 个小样本风格后 p 约 0.05–0.07；只作敏感性/辅助，不单独下结论。
- **Body 仅探索**：22 风格 rho=0.268、p=0.234；n≥5 也仅 0.370/p=0.104。Kaggle 的 Body 词频更接近“厚重/麦芽感”，与官方 `body` 不是同一构念，不能当强证据。
- Salty、Astringency 没有干净官方目标，继续 `do_not_use`。

## 4. 为什么是这组维度：风味结构

5,426 行可用样本的 11 维相关/PCA：

- PC1 解释 **29.2%**：Sour/Fruits 为正，Malty/Body 为负；Sour–Fruits rho **0.824**，Sour–Malty **-0.756**，Fruits–Malty **-0.752**。
- PC2 解释 **22.0%**：Bitter/Hoppy 为正，Sweet/Alcohol 为负；Bitter–Hoppy rho **0.653**。
- PC3 解释 **10.8%**：主要由 Spices 驱动。
- PC4 解释 **9.7%**：Salty/Astringency 等低丰度描述符。
- 其他明显关系：Body–Sour -0.660、Body–Malty 0.607、Body–Fruits -0.607、Bitter–Fruits -0.468、Sweet–Hoppy -0.464、Astringency–Sweet -0.404。

因此：

- Sour 相关最高，是因为它落在 Kaggle 最大的“酸/果香 vs 麦芽/厚重”主轴上，且官方 `acidity` 定义清楚；
- Sweet/Bitter/Malty/Hoppy 对应 PC2 与 PC1 的稳定方向；
- Body 与 Malty 共线，且词频含义偏移，单独拿出来不可靠；
- Spices 由 PC3 支撑，但官方只有 clove/coriander 两个具体香料目标，代理仍是中置信度。

## 5. 映射审计

- 语义映射本身是合理的：Tripel、Kriek、Faro、Flanders old brown、Scotch、Brut、Low/No alcohol 等窄风格基本一一对应；Strong ale、Hoppy、Stout/Porter、Wheat、Pils/Lager 是宽风格多子类合并。
- 我们用 9 维 z-score 做过“最近邻”检查，但结果不可用：由于官方标准化目标与 Kaggle 词频的剖面形状/基线不同，很多官方风格会错误地最近邻到 Low/No alcohol 或 Pils/Lager。**不要用最近邻反向改映射**；映射应继续按风格语义和原始相关性，而不是按剖面余弦。
- 需要降级/单独报告的映射：
  - Faro：官方 n=3、Kaggle 可用 14；只能描述。
  - Brut：官方 n=4、Kaggle 可用 36；只能描述。
  - Kriek：官方 n=10、Kaggle 19（名称/描述子集）；只能描述。
  - Lambic：官方 n=11、Kaggle 89；可作次级。
  - Low/No alcohol：官方 n=8、Kaggle 50；ABV 对照强，但风味画像只作次级。
  - Pils/Lager、Stout/Porter、Hoppy、Strong ale：宽池合并，必须报告“映射敏感性”，不能把单个子风格当成官方风格。
- 映射和代理在正式使用前冻结；冻结后不得因为模型相关性不好而改映射、改合并或改代理目标。

## 6. 对后续外部校验的建议

### Q1 模型外部校验（Q2/Q3 收口后）

- 主检验：Sour、Sweet、Bitter、Malty、Hoppy 五个维度；每个维度在 22 个官方风格上计算“模型预测的风格均值 vs Kaggle 风格画像”的 Spearman。
- 辅助：ABV（强）、Overall rating（中）、Alcohol/Fruits/Spices（次级敏感性）。
- 仅探索：Body；不进入主结论。
- 排除：Salty、Astringency。
- 报告每个维度的 n_styles、官方最小样本、Kaggle `n_flavor_usable`、LOO 范围；Faro/Brut/Kriek/Lambic 单独标注。
- 不把 Kaggle 当训练标签，不参与模型选择，不改变 Q1 主结果。

### Q3 外部描述

- 对 Tripel/Kriek/Lambic 的生成画像，用同一 5 主 + 3 次级维度做方向性描述；
- 只作外部常识旁证，不作生成质量主指标；Kaggle 没有 DMS/双乙酰/4-VG/反-2-壬烯醛浓度，不能用于缺陷约束认证。

### Q2

- 不使用。Kaggle 没有化合物浓度，也没有官方实体键。

## 7. 产物

`docs/analysis/2026初赛/Kaggle深度分析/`

| 文件 | 内容 |
|---|---|
| `kaggle_dimension_consistency.csv` | 9 维 rho、置换 p、LOO、n≥5、推荐角色 |
| `kaggle_numeric_consistency.csv` | ABV、Overall rating 对照 |
| `kaggle_style_level_consistency.csv` | 22 风格的官方/Kaggle 9 维均值、ABV、评分、n |
| `kaggle_style_reliability.csv` | 112 个 Kaggle 风格的 n、可用率、ABV 异常、评分 |
| `kaggle_flavor_spearman.csv` | 11 维风味相关矩阵 |
| `kaggle_flavor_pca_loadings.csv` | 11 维 PCA 载荷（PC1–PC4） |
| `kaggle_analysis_summary.json` | 结论摘要 |

复现：`python scripts/analyze_kaggle_style_consistency.py`

## 8. 结论

这份 Kaggle 数据**可以用**，但只能作为风格级外部合理性证据，而且应聚焦在 **Sour、Sweet、Bitter、Malty、Hoppy + ABV** 上；Alcohol/Fruits/Spices 只作次级，Body 仅探索，Salty/Astringency 不用。映射和代理必须先冻结，再用于 Q1/Q3 收口后的外部校验；不能进入训练、不能当化学浓度或 0–6 真值。

## 9. 第二层审计（2026-10-25）

进一步做了多变量对齐、层级 bootstrap、50×11 子维度扫描和映射形状审计，详见 `docs/analysis/2026初赛/Kaggle多变量对齐与子维度审计.md`：

- 整体对齐为**中等**：Mantel r=0.404（p=0.0004）、RV=0.432（p=0.0016）、Procrustes≈0.436；不是强对齐。
- 1,000 次层级 bootstrap：Sour [0.730,0.921]、ABV [0.661,0.828] 最稳；Sweet/Bitter/Malty/Hoppy 中等；Body [0.144,0.354] 但风格置换 p=0.234，仍只作探索。
- 50×11 扫描有 23 对 q<0.05 且 |rho|≥0.6，但绝大多数是 acidity/Sour/Malty/Body/Fruits 同一条“酸—麦芽”主轴的投影；不能据此扩充主代理。
- 映射形状审计：primary 候选中位排名 6.5/113，9/22 个官方风格排到 20 名以后（Tripel 88、Dubbel 79、Blond 75、Low/No 113）；**不要用最近邻/形状匹配改映射**，语义映射 + 冻结仍是唯一口径。

## 10. Q1 最终模型外部校验（provisional）

用 Q1 v5 最终 NNLS 的 mean-OOF 按官方风格求均值，与 Kaggle 画像比较，详见 `docs/analysis/2026初赛/Kaggle_Q1模型外部校验.md`：

- 5 个主维度全部为正且显著：Sour 0.884、Bitter 0.614、Sweet 0.596、Malty 0.529、Hoppy 0.516；repeat-rho SD 约 0.01–0.03。
- 次级：Alcohol 0.573、Spices 0.557 较强；Fruits 0.392、p=0.070 偏弱；Body 0.335、p=0.134 仅探索。
- Overall rating 从原始 0.494 降到模型 0.294（p=0.178），**不用于外部校验**。
- 这是辅助外部效度，不是独立验证；Q1 v5、Q2/Q3 均未改动，映射/代理仍需冻结。

