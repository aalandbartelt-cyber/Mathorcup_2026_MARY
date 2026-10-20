# MathorCup 比赛项目环境

为 MathorCup 大数据竞赛准备的通用工作环境,与具体赛题数据无关。

## 环境
- conda 环境: `mathorcup` (Python 3.10, torch 2.12+cu128, ultralytics 8.4, opencv 5.0)
- 激活: `conda activate mathorcup`
- R 绘图环境: conda env `r`(ggplot2 4.0 + ragg,见 docs/runbook.md 的 R 章节)
- 已验证: RTX 5070 Ti Laptop GPU 训练可用

## 目录结构
```
mathorcup/
├── data/               # 赛题数据(按 ultralytics 结构放置,见 data/README.md)
├── configs/            # data.yaml 等配置(模板见 data.example.yaml)
├── scripts/            # train.sh / predict_to_csv.py
├── outputs/            # 训练输出、提交文件
├── weights/            # 预训练权重 (yolov8n.pt, yolo26n.pt)
├── figures/            # 论文用图
└── docs/               # 实验记录
```

## 开赛流程
1. 数据放入 `data/<数据集名>/`,按 `configs/data.example.yaml` 生成 `configs/data.yaml`
2. 训练: `bash scripts/train.sh yolov8n 100`
3. 推理出提交文件: `python scripts/predict_to_csv.py --weights <best.pt> --test-dir <测试集目录>`
