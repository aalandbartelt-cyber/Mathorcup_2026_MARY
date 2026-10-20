# 比赛运行手册(Runbook)

> 适用:MathorCup 大数据竞赛。开赛当天按顺序执行，不临时想步骤。

## 环境速查
- conda 环境: `mathorcup`(torch 2.12+cu128 / ultralytics 8.4 / opencv 5)
- 激活: `conda activate mathorcup`; 工作目录: `~/mathorcup`
- 预置权重: `weights/` 下 yolov8n/s/m + yolo26n
- VS Code 打开 `~/mathorcup` → 右侧 Codex 侧栏干活

## 第 0 步:数据清点(0.5h)
1. 数据放 `data/<数据集名>/`,先 `du -sh` 看体积、`ls` 看结构
2. 确认格式:YOLO txt? VOC xml? COCO json? 类别数、是否含负样本
3. 跑 EDA: `python scripts/eda_yolo.py --images ... --labels ... --classes ... --out figures/eda`
4. 记录:类别分布、框尺寸分位、有无空标签/缺标签 → 写进论文"数据分析"节

## 第 1 步:数据准备(0.5h)
1. 划分: `python scripts/split_train_val.py --images data/<名>/images/train --labels data/<名>/labels/train`
2. 复制 `configs/data.example.yaml` → `configs/data.yaml`,改 path 与 names
3. 核对: 每类在 train/val 都有样本; 无数据泄漏(同一图片不得同时出现在两边)

## 第 2 步:Baseline(1~2h)
1. 小跑验证: `bash scripts/train.sh yolov8n 2`(确认链路 + 显存)
2. 正式跑: `bash scripts/train.sh yolov8n 100`
3. 看 `outputs/<run>/results.csv` 的 mAP 曲线判断是否欠训/过训

## 第 3 步:实验矩阵(半天~一天，可并行到自强 A100 四卡)
| 实验 | 目的 |
|---|---|
| yolov8n / s / m | 模型容量 vs 精度/速度 |
| 640 vs 1024 | 小目标/多尺度收益 |
| 默认 vs 关闭 Mosaic | 增强消融 |
| 类别权重/过采样 | 缓解类别不平衡 |
| yolo26n | 新一代模型对比 |

每组实验记录到 `docs/experiment_log.md`: 配置 / mAP50 / mAP50-95 / 各类 AP / 耗时。

## 第 4 步:评估与论文素材(半天)
- 验证: `yolo val model=outputs/<run>/weights/best.pt data=configs/data.yaml` → 各类 AP、PR 曲线、混淆矩阵
- 出图: 预测框可视化、训练曲线、消融对比表
- 论文图统一 300dpi,存 `figures/`

## 第 5 步:提交(最后 2h 内完成，不要拖到截止)
1. `python scripts/predict_to_csv.py --weights <best.pt> --test-dir <测试集> --conf 0.25`
2. **逐项核对赛题格式**: 表头字段、坐标是否归一化、类别 id 映射、无检出行怎么处理
3. 用 `--conf` 扫 0.15/0.25/0.4 看行数变化，选合理阈值
4. 提交前备份 `outputs/test_result.csv` 到 `docs/`

## 常见坑 checklist
- [ ] 数据泄漏: 划分后同图不同名副本、同一视频抽帧
- [ ] 负样本缺失(全图都有目标): 分类问题只能用检测阈值或裁背景 patch
- [ ] 类别不平衡: Hole 类样本少 → 权重/过采样/看 per-class AP
- [ ] 提交坐标: 归一化(0~1) vs 像素(0~W), 中心点+宽高 vs 左上右下
- [ ] 显存 OOM: 降 batch / imgsz; Windows 桌面应用常吃 3GB 显存
- [ ] 路径: 数据/输出放 WSL 原生盘,别用 /mnt/d 跑训练
- [ ] 下载: GitHub 权重在 WSL 直连不稳,用 Windows 代理下载后拷入
- [ ] 中文乱码: 出图前 import scripts/plot_style.py

## 时间分配建议(4 天)
- D1: 读题 + EDA + 数据准备 + baseline
- D2: 主力实验 + 并行消融
- D3: 调优 + 评估 + 出图
- D4: 提交 + 论文成稿(留足半天写论文与检查)
