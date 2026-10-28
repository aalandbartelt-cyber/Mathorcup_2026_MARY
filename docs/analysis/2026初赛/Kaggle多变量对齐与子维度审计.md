# Kaggle 多变量对齐与子维度审计

> 日期：2026-10-25  
> 性质：第二层**原始数据审计**。不使用 Q1/Q2/Q3 模型预测，不训练、不调参、不改变任何主结果。  
> 目的：判断官方 22 风格与 Kaggle 112 风格的整体一致性有多强，50 个官方目标里是否有可靠的细粒度代理，以及“风格最近邻映射”是否可信。

## 1. 方法

- 官方 22 风格 × 9 个代理维度（Body、Alcohol、Bitter、Sweet、Sour、Hoppy、Malty、Fruits、Spices）z-score 后与 Kaggle 官方风格画像比较。
- 整体对齐：
  - **Mantel**：两个风格距离矩阵的相关，5,000 次置换；
  - **RV coefficient**：两个中心化配置矩阵的相似度，5,000 次置换；
  - **Procrustes**：正交对齐后的残差，报告 `1 - disparity`；
  - CCA 不纳入结论：22 风格 × 9 维时极易过拟合，第一典型相关接近 1 也不代表外部效度。
- 不确定性：
  - 对官方数据按风格内啤酒做 **1,000 次层级 bootstrap**，重算风格均值后与固定 Kaggle 画像求 Spearman，报告 95% 百分位区间；
  - 另报 5,000 次风格置换 p 值（第一层报告），两者回答的问题不同。
- 子维度扫描：
  - 官方 50 个目标 × Kaggle 11 个风味维度，在 22 个风格均值上计算 Spearman；
  - 550 个检验做 BH-FDR，报告 q<0.05 且 |rho|≥0.6 的 pair。
- 映射形状审计：
  - 对每个官方风格，用 9 维剖面与 113 个 Kaggle 风格画像逐个计算跨维度 Spearman；
  - 看 primary 映射候选在 113 个候选中的排名；仅作诊断，不用来改映射。

## 2. 整体对齐

| 指标 | 值 | 置换 p | 解释 |
|---|---:|---:|---|
| Mantel r | **0.404** | 0.0004 | 中等正相关，不是强对齐 |
| RV coefficient | **0.432** | 0.0016 | 中等相似度 |
| Procrustes corr（1-disparity） | **0.436** | — | 中等形状一致性 |

结论：两套风格矩阵有可复现的中等整体一致性，但远没有到“同一套画像”的程度。Kaggle 词频只能作为**辅助外部证据**，不能替代官方感官数据，也不能据此说两套数据等价。

## 3. 层级 bootstrap 置信区间

1,000 次风格内啤酒重采样，固定 Kaggle 画像：

| 维度 | 观察 rho | 95% bootstrap CI | 解释 |
|---|---:|---|---|
| Sour | 0.889 | [0.730, 0.921] | 最稳 |
| Sweet | 0.605 | [0.331, 0.717] | 中高，但区间很宽 |
| Bitter | 0.556 | [0.478, 0.654] | 中高，较稳 |
| Malty | 0.519 | [0.362, 0.625] | 中等 |
| Hoppy | 0.510 | [0.405, 0.617] | 中等 |
| Alcohol | 0.490 | [0.386, 0.542] | 中等 |
| Overall | 0.494 | [0.179, 0.616] | 中等但宽 |
| Fruits | 0.447 | [0.129, 0.631] | 方向正，区间很宽 |
| Spices | 0.453 | [0.124, 0.532] | 方向正，区间很宽 |
| Body | 0.268 | [0.144, 0.354] | **仍只作探索**：风格置换 p=0.234，bootstrap 只重采风格内啤酒，不检验风格间可交换性 |
| ABV | 0.775 | [0.661, 0.828] | 强，最可靠的外部数值对照 |

注意：bootstrap CI 不包含 0 不等于“风格间显著”；Body 的置换 p=0.234，仍不能升级为主维度。Sour/ABV 是最稳的两项；Sweet/Bitter/Malty/Hoppy 可作主外部校验；Alcohol/Fruits/Spices/Overall 只作次级。

## 4. 50 × 11 子维度扫描

550 个 pair 中，BH-FDR q<0.05 且 |rho|≥0.6 共 **23 对**。前 15 对：

| 官方目标 | Kaggle 维度 | rho | q |
|---|---|---:|---:|
| acidity | Sour | 0.889 | 1.7e-05 |
| acidity | Malty | -0.834 | 3.9e-04 |
| A_malt_bread | Alcohol | 0.773 | 3.1e-03 |
| acidity | Body | -0.773 | 3.1e-03 |
| acidity | Astringency | 0.770 | 3.1e-03 |
| X4vg | Spices | 0.758 | 3.7e-03 |
| acidity | Fruits | 0.756 | 3.7e-03 |
| A_malt_cara | Astringency | -0.745 | 4.8e-03 |
| acetic | Sour | 0.737 | 5.6e-03 |
| lactic | Sour | 0.731 | 6.1e-03 |
| lactic | Malty | -0.722 | 7.4e-03 |
| acetic | Malty | -0.719 | 7.5e-03 |
| acetic | Bitter | -0.696 | 1.4e-02 |
| lactic | Bitter | -0.679 | 2.0e-02 |
| F_malt_all | Astringency | -0.676 | 2.0e-02 |

解释：

- 绝大多数强相关都在 **acidity / Sour / Malty / Body / Astringency / Fruits** 这条主轴上；它和第一层 PCA 的 PC1（29.2%）一致。这不是 23 个独立发现，而是同一个“酸—麦芽”结构的多种投影。
- `acetic`/`lactic` 与 Sour 的相关（0.737/0.731）支持把 `lactic/acetic` 作为 Sour 的敏感性代理，这与现有代理表一致。
- `X4vg` × Spices（0.758）语义上可解释（4-VG 是香料/酚类风味），但只有一个目标，仍属探索性。
- `A_malt_bread` × Alcohol（0.773）、`F_malt_all` × Astringency（-0.676）很可能是风格混杂或词频共现，不能当新代理。
- **不应根据这张扫描表扩充主代理集**；否则会把风格混杂误当成感官对应。现有 9 维代理已经足够，子维度只作解释和敏感性。

## 5. 映射形状审计

对每个官方风格，用 9 维剖面与 113 个 Kaggle 风格画像逐个算跨维度 Spearman，再看 primary 候选排名：

- 中位 primary 排名：**6.5 / 113**；
- 排名 ≤10：12/22；
- 排名 >20：9/22；
- 典型异常：Tripel 的 primary `Tripel` 排名 88、Dubbel 的 `Dubbel` 排名 79、Blond 的 `Blonde Ale - Belgian` 排名 75、Low/No alcohol 的 `Low Alcohol Beer` 排名 113；这些官方风格的 9 维剖面与 Kaggle 同名词条的剖面形状并不接近。
- 这不是说语义映射错了，而是说 **Kaggle 词频画像的“形状”不能用来做最近邻重映射**。官方标准化目标与 Kaggle 评论词频的基线/构念不同，形状匹配会被主轴和低丰度维度支配。

结论：继续使用语义映射 + 原始相关性证据；不要用最近邻/形状匹配自动改映射。映射和代理在正式模型外部校验前冻结。

## 6. 对后续外部校验的最终建议

1. **整体外部效度**：中等（Mantel/RV≈0.40–0.43），只能作辅助证据，不能写“两套数据高度一致”。
2. **主维度**：Sour、Sweet、Bitter、Malty、Hoppy；加 ABV 作为强数值对照。
3. **次级**：Alcohol、Fruits、Spices、Overall；只作敏感性，不单独下结论。
4. **仅探索**：Body；风格置换 p=0.234，bootstrap CI 不改变这个判断。
5. **不用**：Salty、Astringency。
6. **映射**：语义映射 + 冻结；不要用最近邻/形状审计改映射；Faro/Brut/Kriek/Lambic/Low-No 只作描述或次级。
7. **统计报告**：每个维度同时报 Spearman、风格置换 p、层级 bootstrap CI、n_styles；不要把 bootstrap CI 当作风格间显著性检验。
8. **使用边界**：不训练、不进入模型选择、不当化学浓度/0–6 真值；Q2 不使用；Q1/Q3 只在 Q2/Q3 收口后做辅助外部校验。

## 7. 产物

`docs/analysis/2026初赛/Kaggle深度分析/`

| 文件 | 内容 |
|---|---|
| `kaggle_multivariate_alignment.json` | Mantel、RV、Procrustes |
| `kaggle_bootstrap_ci.csv` | 1,000 次层级 bootstrap 的 95% CI |
| `kaggle_target_flavor_crosscorr.csv` | 50×11 全部 rho/p/q |
| `kaggle_target_flavor_top_pairs.csv` | q<0.05 且 |rho|≥0.6 的 23 对 |
| `kaggle_mapping_shape_audit.csv` | 22 个官方风格的 primary 排名与 top1 |
| `kaggle_deep_summary.json` | 第二层结论摘要 |

复现：`python scripts/deep_analyze_kaggle_alignment.py`

## 8. 结论

这份 Kaggle 数据与官方风格画像有**中等**整体一致性，足够做辅助外部校验，但不足以证明两套数据等价。最可靠的对外维度是 **Sour + ABV**，其次是 Sweet/Bitter/Malty/Hoppy；子维度扫描的 23 个显著 pair 大多由同一条“酸—麦芽”主轴驱动，不能用来扩充主代理；映射形状审计进一步说明不能做最近邻重映射。后续只按冻结的 5 主 + 3 次级 + 1 探索方案使用。
