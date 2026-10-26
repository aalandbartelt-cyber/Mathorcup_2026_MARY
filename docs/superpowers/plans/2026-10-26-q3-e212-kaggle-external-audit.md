# Q3 E212 Kaggle可用性与固定映射外部方向审计

> 用户提供canonical工作区的Kaggle深度分析/公开数据，授权可行路线直接采用。本阶段先采用描述性外部审计，不改冻结生成器，不把已看过的外部数据当独立确认，也不启动缺少真实动作转移数据的RL训练。

## 固定范围

- 只读`/home/mary/mathorcup/data/2026初赛/公开数据/Kaggle_top_beer_information/beer_data_set.csv`及processed中的原语义映射/代理表；原始5558×21，风味11维评论词频，非化学/0—6标签。
- 目标Tripel/Kriek/Lambic；主维度固定Bitter/Sweet/Sour/Hoppy/Malty，次级与Body不临时升格。目标代理分别bitternes/sweetness/acidity/F_hops_all/F_malt_all。
- Tripel按直接Style匹配；Lambic取Traditional+Gueuze；Kriek保持既有Fruit中名称/描述kriek|cherry|cerise子集，不重映射。把本轮实际使用的映射与代理复制到新结果快照并记录原文件哈希，不改canonical原件。
- 分母用11维总词频，132全零行不进入比例均值；每啤酒等权，不按评论词数加权。ABV的0<值≤20只作原有数值过滤诊断，不参与本轮模型训练。
- 复用当前Q3收缩Copula的900×50生成画像及两预测器结果，先验证E205运行清单、完整样本/目标和哈希；同时列官方4.csv真实风格均值作域差异参照。Q1/Q2不重训、不改。

## 冻结统计与限制

每维只比较三组风格对，共15个预定方向；不同域的绝对数值不求MAE/RMSE。Kaggle按目标风格啤酒均值计算差，使用目标三风格涉及的酒厂并集，1000次全局酒厂multinomial cluster bootstrap（种子20261026），保留跨风格同酒厂依赖。报告边际95%区间与有效重采样数；不作15项独立显著性或总体准确率主张。

区间跨0→外部方向不确定；否则与Q3最终NNLS均值差、ElasticNet均值差和官方真实均值差各自比较，保留不一致。Q3合成样本数不是新的真实独立样本；不根据其300条样本生成伪真实置信区间。三个风格不能用跨风格相关p值冒充大样本验证。

## 实现步骤

- [ ] 先新增`02_代码/Q3_风格生成/tests/test_kaggle_external.py`：全零/负数词频、语义映射、Kriek子集、酒厂重采样依赖、不确定方向、绝对尺度不混比；失败后实现`src/kaggle_external.py`。
- [ ] 新增`src/run_e212_kaggle_external.py`：读取/哈希/契约、真实和生成画像完整性、固定15方向、外部证据角色、ABV/IBU字段诊断、结果快照与manifest。完成后只验哈希返回，避免覆盖。
- [ ] 运行测试与审计，检查所有15行（含不一致），报告Q3可采用的外部描述，以及RL/迁移先验/深度生成可用性边界。登记真实数据到外部维度的关系，不编造动作、转移、化学浓度。
- [ ] 更新Q3路线与协作交接，commit/push；不修改canonical已有Kaggle分析、用户未提交文件或Q1/Q2论文。
