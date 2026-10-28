# Kaggle 啤酒数据预处理与外部校验方案

> 日期：2026-10-25  
> 状态：数据准备完成，**尚未进入 Q1/Q2/Q3 任何模型、结果或论文**。  
> 定位：Q2/Q3 收口后，用作风格级外部合理性/外部效度校验的辅助数据；不是训练数据，也不是化学证据。

## 1. 数据与产物

原始数据：`data/2026初赛/公开数据/Kaggle_top_beer_information/beer_data_set.csv`  
规模：5,558 行 × 21 列，112 个 Kaggle 风格。  
SHA-256：`bdfd28f8c75a0da1d86073198aa5341cfe19cabece56827a970915936e0b461a`

处理脚本：`scripts/prepare_kaggle_beer.py`（可重复运行，不训练任何模型）

产物目录：`data/2026初赛/公开数据/Kaggle_top_beer_information/processed/`

| 文件 | 内容 |
|---|---|
| `kaggle_beer_clean.csv` | 5,558 行清洗表；保留原始行，增加风味总量/比例、ABV/rating 标记、Kriek 子集标记 |
| `kaggle_style_profiles.csv` | 112 个 Kaggle 风格 + Kriek 子集，共 113 行；n、11 维平均风味比例、ABV 中位数、评分均值、Top2 风味 |
| `kaggle_official_style_mapping_draft.csv` | 官方 22 风格 ↔ Kaggle 风格候选，84 条；primary_pool=1 共 62 条 |
| `kaggle_official_style_profiles_draft.csv` | 22 个官方风格的 Kaggle 汇总画像（按 primary_pool 合并） |
| `kaggle_sensory_proxy_draft.csv` | Kaggle 11 个粗维度 ↔ 官方 50 目标的代理映射；9 个可代理（5 主、3 次级、1 探索）、2 个明确不用 |
| `kaggle_quality_report.md` | 数据质量、清洗口径、映射与使用边界的机读摘要 |

## 2. 数据质量与清洗口径

- 5,558 行全部保留；`Name` 缺失 2 行；`key` 无重复；`Name` 重复 240 行（key 唯一，保留并标记 `name_duplicate`）。
- 11 个风味字段是评论词频，不是化学浓度或专家 0–6 评分。
- 132 行 11 个描述符全为零：保留在清洗表，但 `flavor_usable=False`，不进入风格画像。
- 风味比例 `prop_* = count / flavor_total`，只在 `flavor_usable=True` 时计算。
- ABV：19 行 ≤0、4 行 >20；`abv_valid = 0 < ABV ≤ 20`，只用于 ABV 汇总，不删行。
- Kriek 子集：从 `Lambic - Fruit` 的 Name/Description 匹配 `kriek|cherry|cerise`，共 19 行；这是名称/描述级子集，不是官方 Kriek 实体连接。
- 不做任何跨表实体连接：官方附件没有名称/酒厂，Kaggle `key` 不是官方 `beer_id`。

## 3. 官方 22 风格映射草案

映射文件：`processed/kaggle_official_style_mapping_draft.csv`

- 84 条候选，其中 primary_pool=1 共 62 条；置信度分 high/medium/low。
- 窄风格多为一一对应：Tripel、Dubbel、Saison、Faro、Kriek（子集）、Flanders old brown、Scotch、Brut、Low/No alcohol 等。
- 宽风格采用多子风格合并：Strong ale（4）、Hoppy（6）、Stout/Porter（14）、Wheat（6）、Pils/Lager（11）等；合并按“每款啤酒等权”，不是按子风格等权。
- Kriek 用 `Lambic - Fruit (Kriek/Cherry/Cerise)` 19 行作为 primary；全部 `Lambic - Fruit` 只作 secondary 敏感性，避免把非 Kriek 水果兰比克混入。
- **这是草案，不是最终映射。** 正式使用前必须冻结；冻结后不得根据模型结果改映射或置信度。

映射覆盖与样本量（primary_pool）：

| 官方风格 | Kaggle 可用行数 | 备注 |
|---|---:|---|
| Tripel | 50 | 直接对应 |
| Kriek | 19 | Kriek/Cherry/Cerise 子集，样本少 |
| Lambic | 89/98 | Traditional 40/49 + Gueuze 49/49 |
| Faro | 14/16 | 样本很少，结论降级 |
| Brut | 36/50 | 14 行全零描述符 |
| Low/No alcohol | 50 | ABV 中位数 0.5，评分均值 2.85 |
| 其余宽风格 | 50–700 | Strong ale/Hoppy/Stout/Porter/Pils-Lager 为合并池 |

## 4. 感官代理：11 个 Kaggle 维度只保留 9 个

官方 50 个目标不是 Kaggle 11 个粗维度的细分版；不能强行 50 维全映射。代理草案：

| Kaggle 维度 | 官方主代理 | 官方敏感性代理 | 置信度 | 状态 |
|---|---|---|---|---|
| Body | `body` | — | high | 使用 |
| Alcohol | `alcohol` | — | high | 使用 |
| Bitter | `bitternes` | — | high | 使用 |
| Sweet | `sweetness` | — | high | 使用 |
| Sour | `acidity` | `lactic` / `acetic` | high | 使用 |
| Hoppy | `F_hops_all` | `A_hops_all` | high | 使用 |
| Malty | `F_malt_all` | `A_malt_all` | high | 使用 |
| Fruits | `F_esters_fruity` | `A_esters_fruity` / `orange` | medium | 使用 |
| Spices | `clove` + `coriander` 平均 | — | medium | 使用 |
| Astringency | — | — | none | **不用**（无干净对应） |
| Salty | — | — | none | **不用**（无官方咸味目标） |

## 4.1 深度分析结论（2026-10-25）

已完成一轮**原始数据一致性审计**（官方 4.csv 风格均值 vs Kaggle 风格画像，不使用任何模型预测），详见 `docs/analysis/2026初赛/Kaggle数据深度分析与映射审计.md`。结论：

- **主外部维度（5 个）**：Sour、Sweet、Bitter、Malty、Hoppy；22 风格 Spearman 为正、置换 p<0.05、留一风格后仍为正。
- **次级维度（3 个）**：Alcohol、Fruits、Spices；方向为正但 n≥5 后 p 约 0.05–0.07，只作敏感性。
- **仅探索（1 个）**：Body；22 风格 rho=0.268/p=0.234，不能当强证据。
- **明确不用（2 个）**：Salty、Astringency；官方数据没有干净对应目标。
- 数值对照：官方 `10**ethanol` 风格中位数 vs Kaggle ABV 中位数 rho=**0.775**（p=0.0002）；官方 overall 风格均值 vs Kaggle 评分 rho=**0.494**（p=0.022）。
- 小样本风格：Faro 官方 3/Kaggle 可用 14，Brut 4/36，Kriek 10/19，Lambic 11/89；这些只作描述或次级。
- 最近邻剖面检查不可靠（官方标准化目标与 Kaggle 词频的剖面形状不同，很多风格会错误最近邻到 Low/No alcohol）；**不要用最近邻反向改映射**，映射继续按风格语义和原始相关性。
- 第二层多变量审计：Mantel r=0.404（p=0.0004）、RV=0.432（p=0.0016）、Procrustes≈0.436，整体只是中等；50×11 子维度扫描的 23 个显著 pair 大多由同一条“酸—麦芽”主轴驱动，不能扩充主代理；映射形状审计 primary 候选中位排名 6.5/113，不能做最近邻重映射。详见 `Kaggle多变量对齐与子维度审计.md`。

## 5. 以后怎么用（Q2/Q3 收口后）

### 5.1 先冻结

1. 冻结 `kaggle_official_style_mapping_draft.csv` 和 `kaggle_sensory_proxy_draft.csv`，记录 SHA-256；
2. 冻结 11 个风味列定义和 primary/secondary 选择；
3. 冻结后再看模型结果；不得因为相关性不好而回头改映射、改风格合并或改代理目标。

### 5.2 Q1 外部风格一致性

- 用 Q1 最终模型（当前为感官组收缩 NNLS）计算每个官方风格的平均预测值；
- 按 5 个主维度（Sour、Sweet、Bitter、Malty、Hoppy）做主要比较；Alcohol/Fruits/Spices 作次级敏感性；Body 仅探索；Salty/Astringency 不用。每个维度在“官方风格 × Kaggle 风格画像”上计算 Spearman 秩相关；
- 报告每个维度的 n_styles、官方最小样本数、Kaggle `n_flavor_usable`；
- 主结果用 `F_hops_all`/`F_malt_all`/`F_esters_fruity`；敏感性用 A 维、`orange`、`lactic/acetic`；
- Faro、Kriek、Lambic Traditional、Brut 等小样本风格单独标注，不纳入强结论或只作描述性展示；
- **不把 Kaggle 当训练标签，不进入模型选择，不改变 Q1 主结果。**

### 5.3 Q3 生成画像的外部描述

- 对 Tripel/Kriek/Lambic 的生成谱，用 Q1 画像器得到 5 个主维度 + 3 个次级维度（Body 仅探索），与 Kaggle 对应风格画像做描述性比较；
- 只作“方向是否常识一致”的外部旁证，不作为生成质量的主指标；
- Kaggle 没有 DMS、双乙酰、4-VG、反-2-壬烯醛等缺陷浓度，不能用于 C1/C2/C3 缺陷约束认证；
- 不把 Kaggle 词频当感官真值，不声称 100% 约束满足。

### 5.4 Q2 不使用

Kaggle 没有化合物浓度，也没有可与官方 `beer_id` 连接的实体键，不能用于 Q2 的 Top-3 化学归因或稳定性证据。

## 6. 合规与引用

- 题面推荐来源：Kaggle “Beer Information - Tasting Profiles”（stephenpolozoff/top-beer-information）。
- Kaggle API 未给出明确许可证；论文按题面推荐来源引用，支撑材料不重新分发原始数据，只保留来源、版本日期和 SHA-256。
- 数据目录已被 `.gitignore` 忽略，不进入公开仓库。

## 7. 产物哈希

| 文件 | SHA-256 |
|---|---|
| `beer_data_set.csv` | `bdfd28f8c75a0da1d86073198aa5341cfe19cabece56827a970915936e0b461a` |
| `kaggle_official_style_mapping_draft.csv` | `f239f8c3670ec3c9b510345691c029fc333d3ce0178edf70675b444930ac45e3` |
| `kaggle_sensory_proxy_draft.csv` | `e77dd37a229d590887b63cc19cc1159e572aa1b05070a9f0afa80e53e71eae8a` |
| `kaggle_official_style_profiles_draft.csv` | `1d2c9933e2fadeda1930d2f397a4f286638d14f75f22ee33edf52225e54d497d` |
| `kaggle_style_profiles.csv` | `f5e4755a3f3f05508f8a8b35f32e76f324e6fbdc2c6301fc0cc78160f2555fcb` |

## 8. 当前结论

数据处理已完成，但**当前不进入任何模型或论文**。等 Q2/Q3 收口后，按冻结映射 + 5 主/3 次级/1 探索代理做一次风格级外部校验即可；不要把它写成训练数据、独立化学验证或 0–6 评分真值。
