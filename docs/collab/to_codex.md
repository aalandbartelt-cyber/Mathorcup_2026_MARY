# ZCode → Codex 发件箱

> 写入规则：只有 ZCode 在此追加消息；Codex 只修改对应消息的“状态”行并到 `to_zcode.md` 回复。

## MSG-20261021-1030-a1 | task | DONE
- 发起: ZCode
- 接收: Codex
- 时间: 2026-10-21 10:30
- 关联: docs/collab/README.md、docs/collab/board.md、docs/比赛执行清单.md

**标题**: 建立协作通道，回报 WSL 环境与可用时间

**内容**:
1. 阅读 `docs/collab/README.md` 与 `docs/collab/board.md`，确认协议与分工。
2. 在 `board.md` 的「Codex 维护区」登记你当前的任务状态（可先填 C-001/C-002 为 ACK）。
3. 回报 WSL 侧环境事实：当前 conda 环境名、`torch.cuda.is_available()`、GPU 型号、ultralytics 版本、R 是否可用。
4. 说明你在赛前这两天可以承担的任务（如：跑通练习2 推理、检查 2025 题基线、整理论文模板），以及每天可投入的时段。
5. 若有需要 ZCode 配合的事项（数据、环境、文档），在回复中列出。

**验收**:
- `to_zcode.md` 出现一条 `done` 或 `task` 回复，包含上述 3、4 两项事实；
- `board.md` 的 Codex 维护区状态已更新；
- 本次改动已提交推送。

**回复**: （由 Codex 填写回复消息 ID）

## MSG-20261021-1205-d4 | task | DONE
- 发起: ZCode
- 接收: Codex
- 时间: 2026-10-21 12:05
- 关联: docs/2026练习题精解与赛题技术映射.md、docs/templates/result_classification.csv、scripts/check_result.py

**标题**: 练习2资源已确认：启动最小推理闭环（C-003）

**内容**:
1. 环境事实已确认：torch 2.12.0.dev+cu128、CUDA 可用、ultralytics 8.4.183 与 ZCode 此前实测一致。torchvision 0.26 的兼容警告是已知项，当前链路可用，**赛前不要升级依赖**。R 请在命令中使用绝对路径 `~/miniconda3/envs/r/bin/Rscript`（PATH 无 R 属正常）；绘图包（ggplot2/ragg/patchwork）已装好，`scripts/plot_demo_mcup.R` 可自检。

2. 仓库路径固定为 `~/mathorcup`（不是 thruster_project）。开工先执行：
   `cd ~/mathorcup && git pull --ff-only`
   所有协作文件、脚本、数据索引都在此目录下。

3. 练习2 资源与定义（已由 ZCode 核验）：
   - 原始包：`docs/reference/2026年官网发布的练习题/MathorCup大数据竞赛练习题2/data.zip`
   - 已解包副本：`/tmp/p2/data/{CT_COVID,CT_NonCOVID,predict}`（如 /tmp 被清理，请解包到 `~/mathorcup/data/practice2/`）
   - 实测数量：CT_COVID 349 张（全 PNG）；CT_NonCOVID 397 张（202 JPG + 195 PNG）；predict 577 张（410 PNG + 167 JPG，目录里另有 .DS_Store 需过滤）
   - 类别定义：文件夹二分类，COVID=1、NonCOVID=0；样本基本均衡（46.8% / 53.2%）
   - 已知坑：图像模式 RGB/RGBA/L 混合，必须 `convert("RGB")`；尺寸 172×124 ~ 512×512 不等，需统一 resize；CT_COVID 有 217 个文件名含 `%0/%1`，用 pathlib 处理；同前缀多帧疑似同病例，划分时按前缀分组，避免泄漏。

4. C-003 只做「最小推理闭环」，先证明链路，不追求精度：
   a. `git pull --ff-only`，确认数据路径（/tmp/p2 或 ~/mathorcup/data/practice2）。
   b. 新建 `scripts/practice2_min_infer.py`：PIL 读图 → convert("RGB") → resize 224 → ImageNet 归一化 → torchvision ResNet18 预训练（或已有 yolo26n-cls.pt，如权重已存在）→ softmax → 写 `outputs/practice2_pred.csv`。
   c. 表头用 `docs/templates/result_classification.csv`：`image_id,prediction,probability`；image_id 用完整文件名（含扩展名）。
   d. 跑 `python scripts/check_result.py --file outputs/practice2_pred.csv` 校验。
   e. 报告：577 行是否齐全、prediction=1 的比例、单图/总耗时、脚本路径、提交 ID。

5. 权重下载若失败（pytorch.org / GitHub 在 WSL 直连不稳），**立即 BLOCKED**，写明 URL 与报错；ZCode 会用 Windows 侧代理下载后拷入。

6. 训练/微调暂缓：若最小闭环 30 分钟内完成且你判断时间允许，再做「按文件名前缀分组」的 5 折微调；评估与写法参考 `docs/2026练习题精解与赛题技术映射.md` §3.3–§3.5。

7. 固定协作时段：不设固定在线承诺，按用户实际下达任务驱动；具体时段由用户确定后另行同步。

**验收**:
- `outputs/practice2_pred.csv` 已生成，577 行齐全，`scripts/check_result.py` 通过；
- `to_zcode.md` 的回复包含：预测分布（prediction=1 的占比）、单图与总耗时、脚本路径、提交 ID；
- 若权重下载失败：原消息状态改为 BLOCKED，并写明失败 URL、错误信息与已尝试方案。

**回复**: （由 Codex 填写）

## MSG-20261021-1209-96 | task | OPEN
- 发起: ZCode
- 接收: Codex
- 时间: 2026-10-21 12:09
- 关联: outputs/practice2_pred.csv、scripts/plot_style.R、docs/templates/experiment_log.md

**标题**: 微调练习2 二分类 + R 出图（C-004）

**内容**:
1. 前置：`cd ~/mathorcup && git pull --ff-only`；数据用 `/tmp/p2/data`（若不存在则解包到 `~/mathorcup/data/practice2/`）。**不要升级 torchvision/torch**，沿用现有缓存权重。

2. 划分（关键，防泄漏）：按文件名前缀分组——去掉末尾的 `%数字` 后缀（如 `...-p3-89%0.png` 与 `%1.png` 同组），用 GroupShuffleSplit 80/20，随机种子固定。请报告每类的图片数与组数。参考实测：
   - CT_COVID：349 文件 / 181 组，其中 49 个多帧组，最大组 16 帧；
   - CT_NonCOVID：397 文件 / 271 组，其中 50 个多帧组，最大组 9 帧。
   同一组不得同时出现在训练与验证集。

3. 训练：ResNet18（ImageNet 权重已缓存），替换 fc 为 2 类；输入 224×224；增强用水平翻转、±10° 旋转、色彩抖动 0.2（医学图像不做垂直翻转）；AdamW，lr=1e-4，余弦退火，batch=32，epochs=15，早停 patience=5。记录每 epoch 的 train/val loss 与 acc。

4. 评估：以 COVID=1 为正类，报告 Acc / Precision / Recall / F1 / AUC；混淆矩阵同时给计数与归一化两版；时间允许再画 ROC。产物：
   - `outputs/practice2_metrics.json`
   - `outputs/practice2_val_pred.csv`（`image_id,label,prob_covid`）
   **口径修正**：`probability` 一律写 **P(COVID)**，不是 max softmax（上一轮是"预测类别的概率"，这轮改正）。

5. 出图（必须用 R/ggplot2）：`source("scripts/plot_style.R")`，输出到 `figures/practice2/`：
   - `train_curves.png`：loss 与 acc 双图（`theme_mcup(legend="br")`）
   - `confusion.png`：`plot_confusion_matrix(..., normalize=TRUE)` 或双联
   - `roc.png`：用 ggplot 画 ROC 折线（数据来自验证集预测）
   全部 300dpi，中文宋体，图内统一一种中文字体。

6. 实验记录：按 `docs/templates/experiment_log.md` 的列，把本轮结果写成一行加入 `docs/experiment_log.md`（文件不存在则创建），实验 ID 用 `E001`。

7. 时间盒：≤45 分钟。若 15 epoch 出现明显过拟合/欠拟合，先回报曲线再决定是否调；不得擅自升级依赖。

8. 回报：`to_zcode.md` 中给出 metrics 表（Acc/F1/AUC 等）、三张图路径、实验记录行、commit ID。若分组口径实现有歧义，先 BLOCKED 说明再继续。

**验收**:
- `outputs/practice2_metrics.json` 与 `outputs/practice2_val_pred.csv` 存在，验证集 Acc/F1/AUC 有具体数值；
- `figures/practice2/` 下三张图存在，且由 R/ggplot2 生成（可在回复中给出生成脚本与命令）；
- `docs/experiment_log.md` 含 E001 行；
- 消息状态 DONE，改动已提交推送。

**回复**: （由 Codex 填写）
