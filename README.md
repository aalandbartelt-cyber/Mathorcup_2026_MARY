# MathorCup 2026 比赛项目环境

2026 初赛已锁定 A 题。本仓库按“赛题与规则—数据—代码—结果—图表—论文—交付”组织；每一问在 `02_代码/` 与 `05_论文/` 中都有同名目录，避免源码、实验结果和论文材料混放。

当前核心入口：

- Q1 技术分析：`docs/analysis/2026初赛/A题第一问详细分析.md`
- Q1 源码：`02_代码/Q1_感官评分/`
- Q1 论文交接：`05_论文/Q1_感官评分/`
- 实验记录：`docs/experiment_log.md`
- 最终交付：`06_交付/`

## 环境清单

| 环境 | 内容 | 验证状态 |
|---|---|---|
| Python: conda `mathorcup` | Python 3.10 / torch 2.12+cu128 / ultralytics 8.4 / opencv 5 / sklearn / xgboost / lightgbm / optuna | ✅ GPU 训练与推理通过 |
| R: conda `r` | R 4.5.3 / ggplot2 4.0 / dplyr / data.table / patchwork / ragg / languageserver | ✅ 中文出图通过 |
| 出图字体 | 英文 Times New Roman，中文宋体 SimSun（黑体 SimHei 备选），300dpi | ✅ 实测无缺字 |
| 预训练权重 | `weights/`：yolov8n / yolov8s / yolov8m / yolo26n | ✅ |

- Python 激活：`conda activate mathorcup`；R 激活：`conda activate r`
- 出图：固定 R/ggplot2（`scripts/plot_style.R`，参考图 `figures/demo_m_cup_style_R.png`）；Python/matplotlib 版保底
- GPU：RTX 5070 Ti Laptop（12GB，Blackwell sm_120），WSL GPU 直通

## 协作流（设备与分工）

| 角色 | 所在位置 | 职责 |
|---|---|---|
| **Codex** | WSL 内（VS Code Remote-WSL 的 Codex 扩展） | **核心开发**：训练/推理/评估代码、实验执行、调试、LaTeX 编译 |
| **ZCode** | Windows 宿主（ZCode 桌面端，运行 DeepSeek/GLM Flash 等轻量模型） | **辅助**：环境搭建与运维、数据整理、代码审查、结果交叉验证、Codex 额度不足时替补 |
| **自强平台 4×A100** | 校园 HPC（网页终端 + 云盘） | **重训练与并行消融**：每卡独立跑一个实验（不用 DDP，数据量小 DDP 收益差） |
| **本地 5070 Ti** | WSL GPU 直通 | 调试、小实验、测试集推理、出图、生成提交 CSV |

### 工作流

```
本地 WSL 调试（Codex） → 小规模验证通过
        │
        ├─ git push ──→ 自强平台 git pull ──→ 4×A100 并行正式实验
        │                                       │
        │                                       └─ 指标回填 docs/experiment_log.md
        │
        └─ 测试集推理 + 论文图表（本地完成）
```

### 分工原则

- **Codex 离环境和数据最近**：它直接住在 WSL 里改文件、跑训练，零转换层。
- **ZCode 负责"环境与质量的兜底"**：装环境、修网络、整理数据、审 AI 输出的错（数据泄漏、指标口径、提交格式），不抢核心开发。
- **算力按任务分配**：调试和出图不占 A100；正式训练和消融在 A100 四卡并行；本地 GPU 只做需要快速反馈的事。
- **GPU 不影响精度只影响速度**：最终 mAP 取决于模型/数据/增强策略，不取决于用哪张卡。

## 数据与产物流动

| 内容 | 位置 | 是否进 git |
|---|---|---|
| 分问题代码 | `02_代码/Q1_感官评分/` 等 | ✅ |
| 赛题数据 | `data/<数据集名>/` | ❌（`.gitignore`） |
| 预训练权重 | `weights/*.pt` | ❌（体积大，按需下载） |
| 可复现实验结果 | `03_结果/<问题>/` | ✅（大模型权重除外） |
| 论文图 | `04_图表/<问题>/` | ✅（体积小，便于共享） |
| 论文工作稿与交接材料 | `05_论文/<问题>/` | ✅ |
| 最终提交文件 | `06_交付/` | 按类型管理 |
| 往年赛题与优秀论文 | `docs/reference/` | ❌（版权与体积） |
| 实验记录 | `docs/experiment_log.md` | ✅ |

平台数据流转：自强平台的数据通过云盘单独上传，不经过 git；代码用 git 同步。

## 目录结构

```
mathorcup/
├── 00_赛题与规则/       # 赛事参数、选题结论、规则索引
├── 01_数据/             # 数据契约、对账表、固定划分与派生表
├── 02_代码/             # Q1/Q2/Q3 源码与配置
├── 03_结果/             # Q1/Q2/Q3 可复现实验结果
├── 04_图表/             # Q1/Q2/Q3 最终论文图
├── 05_论文/             # 正文源文件与逐问论文交接包
├── 06_交付/             # 最终论文、结果、承诺书、支撑材料
├── data/                # 原始赛题附件，只读且不进 git
├── scripts/             # 通用兼容脚本与 R 绘图样式
├── docs/                # 研究记录、协作、参考资料和实验日志
├── _archive/            # 旧方案与废弃版本
├── outputs/ figures/    # 旧流程兼容目录，不再放新问题产物
├── weights/             # 预训练权重
└── README.md
```

`data/`、`scripts/`、`outputs/`、`figures/` 为早期通用流程保留，暂不移动以避免已有命令失效。2026 A 题新增文件统一使用编号目录。

## AI 工具合规（2026 新规）

- 规定全文与落地清单：`docs/AI工具使用规定_2026.md`
- 论文声明/参考文献/正文标注/代码注释模板：`docs/templates/paper_ai_declaration.md`
- 支撑材料：`06_交付/05_AI工具使用说明/AI工具使用详情说明.pdf`
- 要点：参考文献后声明、参考文献列 AI 工具、正文/图表标注、代码注释头、支撑材料 PDF，缺一取消资格。

## 自强平台（A100）

上 A100 前先读 `docs/自强平台操作指南.md`：容器与云端持久盘、逐卡自检、双卡只有一张可用时的处理、
GitHub DNS 故障时用 patch 同步、长任务保活与状态脚本、CRLF/`watch` 段错误等报错速查、收尾停容器。

## 2026 提交（MCB2601605）

- 队号 **MCB2601605** → 文件名 **A2601605 / B2601605**（题号 + 7 位数字）
- 提交区：`06_交付/`（00 官方模板、01 论文、02 结果、03 承诺书、04 支撑材料）
- 规范速查：`docs/2026提交规范速查.md`；逐项清单：`06_交付/提交清单.md`
- 归档改名：`python scripts/prepare_submission.py --root . --team MCB2601605 --track A --paper ... --result ...`
- 提交入口：https://www.saikr.com/vse/bigdata2026

## AI 协作（ZCode ↔ Codex）

三个角色通过仓库内的文件信箱交接：**Codex**（WSL，核心：建模/选型/结果解读/实现）、**ZCode**（Windows，环境/复核）、**dsh（DeepSeek Harness，Windows，论文评审与优化）**。协议见 `docs/collab/README.md`：

- `docs/collab/board.md` — 任务板（当前任务/状态/阻塞/日志）
- `docs/collab/to_codex.md` — ZCode → Codex（Codex 只改状态行）
- `docs/collab/to_zcode.md` — Codex → ZCode（ZCode 只改状态行）
- 追加消息：`python scripts/collab_msg.py --to codex --kind task --title "..." --body "..." --accept "..."`
- 给 Codex 的启动指令见协议 §0（复制粘贴即可）

## 赛前 48 小时清单（开赛 2026-10-23 18:00）

打印版：`docs/比赛执行清单.md`（含官方答卷六大问题自查表、结果文件模板说明、提交前 20 项终检）。

```bash
# 1) 结果文件校验（提交前必跑）
python scripts/check_result.py --file outputs/result_detection.csv --images-dir data/<数据集>/images/test
python scripts/check_result.py --file outputs/result_classification.csv

# 2) 环境快照（写论文附录/复现说明用）
Rscript scripts/session_info.R          # → docs/env_r.txt
pip freeze > docs/env_py.txt

# 3) 结果模板
ls docs/templates/                      # 分类 CSV / 检测 CSV / 分配 TXT / 实验记录表
```

## 开赛流程

1. 数据放入 `data/<数据集名>/`，按 `configs/data.example.yaml` 生成 `configs/data.yaml`
2. EDA：`python scripts/eda_yolo.py --images ... --labels ... --classes ...`
3. 划分：`python scripts/split_train_val.py --images ... --labels ...`
4. 训练：`bash scripts/train.sh yolov8n 100`
5. 推理出提交文件：`python scripts/predict_to_csv.py --weights <best.pt> --test-dir <测试集>`
6. 核对提交格式（表头、坐标归一化、类别映射）后提交

四天赛程与坑清单详见 `docs/runbook.md`。

## 已知环境注意事项

- **代理**：主力用 SpeedCloud（固定 `127.0.0.1:7892`）。不要使用 v2rayN 界面里的 `10808`——该端口从未被内核绑定，写死它的配置（如 VS Code `http.proxy`）都会失败。
- **长训练**一律在 `tmux` 内跑，断线/关窗不中断。
- **同一时间只让一个 AI 修改同一份文件**（Codex 与 ZCode 并行时），避免冲突。
- **WSL DNS**：部分域名在 WSL 内解析被污染；需要外部资源的任务优先在 Windows 侧下载后拷入。
