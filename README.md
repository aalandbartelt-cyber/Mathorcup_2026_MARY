# MathorCup 2026 比赛项目环境

为 MathorCup 大数据竞赛准备的通用工作环境与协作流程，与具体赛题数据无关。

## 环境清单

| 环境 | 内容 | 验证状态 |
|---|---|---|
| Python: conda `mathorcup` | Python 3.10 / torch 2.12+cu128 / ultralytics 8.4 / opencv 5 / sklearn / xgboost / lightgbm / optuna | ✅ GPU 训练与推理通过 |
| R: conda `r` | R 4.5.3 / ggplot2 4.0 / dplyr / data.table / patchwork / ragg / languageserver | ✅ 中文出图通过 |
| 出图字体 | 英文 Times New Roman，中文宋体 SimSun（黑体 SimHei 备选），300dpi | ✅ 实测无缺字 |
| 预训练权重 | `weights/`：yolov8n / yolov8s / yolov8m / yolo26n | ✅ |

- Python 激活：`conda activate mathorcup`；R 激活：`conda activate r`
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
| 代码 | 本仓库 | ✅ |
| 赛题数据 | `data/<数据集名>/` | ❌（`.gitignore`） |
| 预训练权重 | `weights/*.pt` | ❌（体积大，按需下载） |
| 训练产物/提交文件 | `outputs/` | ❌ |
| 论文图 | `figures/` | ✅（体积小，便于共享） |
| 往年赛题与优秀论文 | `docs/reference/` | ❌（版权与体积） |
| 实验记录 | `docs/experiment_log.md` | ✅ |

平台数据流转：自强平台的数据通过云盘单独上传，不经过 git；代码用 git 同步。

## 目录结构

```
mathorcup/
├── data/                 # 赛题数据（结构见 data/README.md）
├── configs/              # data.yaml（模板 data.example.yaml）
├── scripts/
│   ├── eda_yolo.py       # 通用 EDA，出 4 张论文级图
│   ├── split_train_val.py# 分层 train/val 划分
│   ├── train.sh          # 一键训练
│   ├── predict_to_csv.py # 测试集推理 → 提交格式 CSV
│   ├── plot_style.py     # matplotlib 论文样式（Times+宋体）
│   ├── plot_style.R      # ggplot2 论文样式（宋体/黑体）
│   ├── plot_example.R    # R 出图示例
│   └── setup_r.sh        # R 环境重建
├── outputs/              # 训练输出、test_result.csv
├── weights/              # 预训练权重
├── figures/              # 论文用图
├── docs/
│   ├── runbook.md        # 比赛运行手册（四天流程+坑清单）
│   └── reference/        # 往年赛题/优秀论文/2026练习题
├── requirements.txt      # Python 依赖快照
└── README.md
```

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
