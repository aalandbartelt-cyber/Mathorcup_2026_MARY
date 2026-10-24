# Task Plan: 参考方案对照与 Q1/Q2 优化交付

## Goal

在不复制 BZD/agent 文字和代码的前提下，把参考方案转化为可验证的候选实验；以固定 OOF 口径决定是否替换 Q1 v1，并按检查点完成 Q1 图稿、Word v2、交付包及 Q2 优化。

## Current Phase

Phase 2B：Q1 图表 V3 证据层级精修设计已完成，等待规格确认后实施。

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
- [ ] 等待用户确认
- **Status:** waiting for user review

### Phase 2B：Q1 图稿 V3 精修

- [x] 对照 BZD、agent 与 615 张优秀论文风格统计，完成视觉审计
- [x] 冻结 V3 图1—图4与附图 S1 的图形编码设计
- [x] 写入并自检 V3 设计规格
- [ ] 用户确认书面规格
- [ ] 生成 `final_v3_draft/`，完成缩放、灰度和重绘一致性检查
- [ ] 向 `to_zcode.md` 发送 V3 图清单，等待最终验图
- **Status:** waiting for design approval

### Phase 3：Q1 正文 v2 与打包

- [ ] 用户确认图后，复制 agent Q1.docx 作为版式模板
- [ ] 用本队内容重建“问题一分析—模型—求解/检验”
- [ ] 公式使用 Word 原生 OMML；图题在 Word 中，不嵌入图片
- [ ] 运行 handoff 脚本并核对交付包
- **Status:** pending user figure confirmation

### Phase 4：Q1 候选优化

- [ ] 累积有序 HGB
- [ ] 风格中心化 Ridge
- [ ] 折内目标图正则多任务 Ridge
- [ ] 按 10×3 固定 OOF 成对比较，至少两项稳定改善才替换 v1
- **Status:** pending

### Phase 5：Q2 优化与交付

- [ ] 把 BZD 共表达/稳定选择与 agent 双路线思想映射到现有 E105—E112
- [ ] 保持折内重算、50×3 完整输出和化学证据分级
- [ ] 生成 Q2 Word 与交付包
- **Status:** pending

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

