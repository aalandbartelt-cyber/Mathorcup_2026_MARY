# Task Plan: 参考方案对照与 Q1/Q2 优化交付

## Goal

在不复制 BZD/agent 文字和代码的前提下，把参考方案转化为可验证的候选实验；以固定 OOF 口径决定是否替换 Q1 v1，并按检查点完成 Q1 图稿、Word v2、交付包及 Q2 优化。

## Current Phase

Phase 6：Q2 结果冻结后的论文 v3 质量升级已完成，保留 v2 并完成 DOCX/PDF/ZIP 审计。

## Phases

### Phase 1：参考材料审计与指标对齐

- [x] 阅读 BZD AA/AB/AC 论文、代码、图和方案对比表
- [x] 阅读 agent Q1/Q2/Q3 代码、日志、图和 Word
- [x] 阅读新 skill 的 01/02/03/05/06/13/14/15
- [x] 为 E003 补算 Spearman、代理 QWK、代理 ACC±1
- [x] 形成 `BZD与agent对照.md`
- **Status:** complete

### Phase 2：Q1 图稿 v2 检查点

- [x] 将图 2、图 4 的轴、图例与注释改为中文呈现层
- [x] 保留 `final_v1`，新稿另存 v2，不把总图题画入图片
- [x] 300 dpi 导出并做尺寸、字体、遮挡审计
- [x] 向 `to_zcode.md` 发送图清单
- [x] 用户已指出图1灰度过重、图2版式不佳，v2 不再作为当前候选
- **Status:** superseded by V3 review draft

### Phase 2B：Q1 图稿 V3 精修

- [x] 对照 BZD、agent 与 615 张优秀论文风格统计，完成视觉审计
- [x] 冻结 V3 图1—图4与附图 S1 的图形编码设计
- [x] 写入并自检 V3 设计规格
- [x] 用户确认书面规格并授权直接重画
- [x] 根据首轮反馈把图1改为低饱和多色，把图2重构为三栏 R²—Spearman 性能图谱
- [x] 根据后续反馈移除图2标题计数、横排图4总体 MAE、恢复图4(b)底部图例并简化图1数值标注
- [x] 生成 `final_v3_draft/`，完成缩放、灰度和重绘一致性检查
- [x] 向 `to_zcode.md` 发送 V3 图清单，等待最终验图
- **Status:** waiting for final figure review

### Phase 3：Q1 正文 v2 与打包

- [ ] 用户确认图后，复制 agent Q1.docx 作为版式模板
- [ ] 用本队内容重建“问题一分析—模型—求解/检验”
- [ ] 公式使用 Word 原生 OMML；图题在 Word 中，不嵌入图片
- [ ] 运行 handoff 脚本并核对交付包
- **Status:** pending user figure confirmation

### Phase 4：Q1 候选优化

- [x] 冻结五指标全部冲击的硬门槛
- [x] 确认双路线嵌套 OOF、连续集成与收缩门控架构
- [x] 写入第二轮优化设计规格并完成规格自检
- [x] 用户复核书面规格
- [x] 生成逐任务 TDD 实施计划
- [ ] 按实施计划内联实现 E004—E007
- [ ] 按筛查 3 重复 → 正式 10 重复完成晋级判断
- **Status:** design checkpoint

### Phase 5：Q2 优化与交付

- [x] 把公开方法家族映射到现有稳健统计与独立归因证据
- [x] 保持折内重算、50×3 完整输出和化学证据分级
- [x] 生成 Q2 Word 与交付包；E112 仍为正式答案，E115 为独立解释证据
- **Status:** complete; numeric result frozen

### Phase 6：Q2 论文 v3 质量升级

- [x] 冻结 v2 哈希与 v3 结果边界
- [x] 完成 v3 设计规格和实施计划
- [x] 新增 OMML、图像、禁用词和 v2 不变性检查
- [x] 重建 7 个原生 OMML 公式
- [x] 用 R/ggplot2 重绘 4 张 3:2 主图
- [x] 扩写公开方法对照、消融、摘要式开篇与讨论
- [x] 生成 v3 DOCX/PDF/审计合订本/ZIP
- [x] 运行 119 项既有测试、新增检查及全量扫描
- **Status:** complete; 125 tests passed, 7 subtests passed

## Hard Rules

| 规则 | 执行口径 |
|---|---|
| 不直接复制 | 参考材料只用于候选设计、风险识别与结果核对 |
| 防泄漏 | 插补、缩放、分箱、构图、筛选和选参全部在训练折内 |
| 代理等级诚实命名 | BZD 式 QWK/ACC±1只称“独立分位数分箱代理指标” |
| 图稿 | R/ggplot2 `theme_mcup()`；无内嵌总图题；面板号加粗；呈现层中文 |
| Word | 使用指定 agent 分问模板版式；数学公式必须为原生 OMML |
| 模型替换 | R²/Spearman/代理 QWK/代理 ACC±1至少两项在多数重复中稳定改善 |
| 提交节奏 | 对照、图稿、Word/打包、Q2 分别 commit/push |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| 默认 Python 缺 numpy/pandas | 1 | 统一使用 `conda run -n mathorcup` |
| `conda run` 通过 stdin 不回显 | 1 | 改用脚本文件或 `python -c` |
| 参考 QWK 被误读为真实 0—6 | 1 | 追到 BZD `quantile_levels` 源码，明确为真实/预测各自全样本分箱 |
| 长实施计划首次补丁的 Markdown 命令块缺少补丁前缀 | 1 | 首次写入未产生文件；改用逐行生成补丁并完成占位与格式扫描 |
| 规格状态行保留 Markdown 行尾双空格导致 `git diff --check` 失败 | 1 | 删除行尾空格后重新执行提交前检查 |
| pytest 自动加载 ROS `launch_testing`，缺少 `lark` | 1 | 设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，隔离无关系统插件 |
| 禁用词正则把 SHA256 中偶然出现的 `e100` 当作实验编号 | 1 | 改为字母数字边界识别，仍能检出文件名/正文中的独立编号 |
| ggplot2 4.0 报 `label.size` 与 `geom_errorbarh()` 弃用警告 | 1 | 改用 `linewidth` 与带 `orientation="y"` 的 `geom_errorbar()` |
| 严格 OMML 生成器拒绝公式3中的纯空格 run | 1 | 删除仅用于视觉间距的空白 run，交由 Office Math 结构排版 |

