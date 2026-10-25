# Q2 Paper v3 Quality Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结 Q2 数字与排序的前提下，交付公式、图表、公开方法对照和审计均达标的 v3 DOCX/PDF/ZIP。

**Architecture:** 先把既有结果导出为不含内部代号的论文级 CSV，再由独立 R 脚本生成四图；Word 构建器复制指定模板，调用严格 OMML 生成器插入七式并嵌入 v3 图。独立审计脚本检查公式结构、禁用词、图像规格、模板继承和 ZIP 完整性。

**Tech Stack:** Python 3.11、python-docx/lxml、R 4.x/ggplot2/patchwork、LibreOffice、Ghostscript、pytest。

---

### Task 1: 冻结 v2 与新增失败检查

**Files:**
- Create: `02_代码/Q2_关键化合物/tests/test_q2_paper_v3.py`
- Create: `03_结果/Q2_关键化合物/publication_audit/v2_frozen_sha256.txt`

- [ ] 记录 v2 三个关键产物 SHA256。
- [ ] 新增测试：主 DOCX 七个 OMML、空 `m:r` 为零、每式结构节点和关键符号齐全、Cambria Math 存在。
- [ ] 新增测试：四图均为 2700×1800 或等比例 3:2、约 300 dpi，图源/文件名无禁用词。
- [ ] 新增测试：v3 文档与 ZIP 禁用词为零，v2 哈希未变。
- [ ] 运行专项测试并确认因 v3 尚不存在而失败。

### Task 2: 重构七个 OMML 公式

**Files:**
- Create: `02_代码/Q2_关键化合物/src/q2_omml.py`
- Modify: `02_代码/Q2_关键化合物/src/build_q2_current_handoff.py`

- [ ] 实现非空数学 run、Cambria Math、变量斜体、函数正体的 OMML 原语。
- [ ] 用结构化分式、求和、上下标和定界符重建七式，特别是第六式的 `π_kj`、`1/B`、`Σ_{b=1}^{B}`、`I(k∈T_b^(j))`。
- [ ] 把版本输出参数化为 v3，同时保留 v2 文件不覆盖。
- [ ] 生成临时 v3 DOCX并运行 OMML 专项测试。

### Task 3: 冻结论文级图源并用 R 重绘

**Files:**
- Create: `04_图表/Q2_关键化合物/v3/source_data/*.csv`
- Create: `04_图表/Q2_关键化合物/v3/source_data_manifest.json`
- Create: `04_图表/Q2_关键化合物/v3/figure_contract.md`
- Create: `04_图表/Q2_关键化合物/v3/plot_q2_paper_v3.R`
- Create: `04_图表/Q2_关键化合物/v3/Q2_图清单_v3.csv`

- [ ] 只读导出冻结结果到无内部代号的 CSV，并记录源文件哈希。
- [ ] 按设计生成四张 9×6 英寸、300 dpi 主图，不写整图标题。
- [ ] 生成 70% 与灰度检查图，运行尺寸/DPI/禁用词测试。
- [ ] 逐图目检标签、图例、N/A、单位和数据重叠。

### Task 4: 扩写公开方法对照与同步正文

**Files:**
- Modify: `02_代码/Q2_关键化合物/src/build_q2_current_handoff.py`
- Modify: `05_论文/Q2_关键化合物/README.md`
- Modify: `05_论文/Q2_关键化合物/05_图表与表格清单.md`

- [ ] 更新开篇、分析、结果和讨论，保持全部数字与排序不变。
- [ ] 新增四列表格，对照相关网络、graphical lasso、条件置换/SHAP和稳定选择；引用作者/年份并标明跨任务不可直接横比。
- [ ] 将三证据融合写为未采用消融，把正式 Top-3 与最终感官模型归因明确分层。
- [ ] 更新图题、表题和正文引用，插入 v3 图。

### Task 5: 生成 DOCX/PDF 与逐式目检

**Files:**
- Create: `05_论文/Q2_关键化合物/第二问_论文正文_v3.docx`
- Create: `05_论文/Q2_关键化合物/模型假设_Q2_v3.docx`
- Create: `05_论文/Q2_关键化合物/问题分析_Q2_v3.docx`
- Create: 对应 PDF 与 `Q2_论文审计合订本_v3.pdf`

- [ ] 用 LibreOffice 生成三个 PDF并合并审计本。
- [ ] 用 DOCX 验证器检查 OOXML，确认空 OMML run 为零。
- [ ] 渲染 PDF 页面和公式页，逐式核对分式、求和、上下标、范数与编号。
- [ ] 逐页核对图题、表题、正文引用和版式。

### Task 6: 打包、全量扫描与回归测试

**Files:**
- Create: `05_论文/交付给论文手/Q2_关键化合物_交付包_v3.zip`
- Create: `03_结果/Q2_关键化合物/publication_audit/q2_v3_audit.json`

- [ ] 运行 119 项既有测试和新增检查。
- [ ] 扫描 v3 文档、PDF、图源、文件名及 ZIP 解包内容，禁用词必须零命中。
- [ ] 核对 ZIP 清单、文件数、大小和 SHA256；复核 v2 哈希不变。
- [ ] 运行 `git diff --check` 和交付前验证，提交并推送。
