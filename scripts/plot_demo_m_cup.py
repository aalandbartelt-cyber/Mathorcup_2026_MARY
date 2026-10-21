#!/usr/bin/env python3
"""M 杯出图风格示例：2x2 组合(训练曲线/混淆矩阵/PR曲线/特征重要性)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
import plot_style as ps

ps.set_style("m杯")
rng = np.random.default_rng(0)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# 1) 训练曲线（四模型，同 2023-2 风格）
ep = np.arange(1, 51)
for i, name in enumerate(["DenseNet121", "VGG16", "GoogLeNet", "ResNet50"]):
    axes[0, 0].plot(ep, 0.86 + 0.13 * (1 - np.exp(-ep / 7)) + 0.004 * rng.standard_normal(50),
                    color=ps.CYCLE[i], lw=1.2, label=name)
axes[0, 0].set_xlabel("迭代次数"); axes[0, 0].set_ylabel("准确率")
axes[0, 0].set_ylim(0.70, 1.01); axes[0, 0].legend(loc="lower right")
axes[0, 0].set_title("模型准确率随迭代次数变化")

# 2) 混淆矩阵（Blues + 数值）
cm = np.array([[409, 4], [13, 400]])
ps.plot_confusion_matrix(cm, ["Normal", "Fault"], ax=axes[0, 1], title="混淆矩阵")

# 3) PR 曲线（各类 + 粗体总曲线）
rec = np.linspace(0, 1, 120)
ps.plot_pr_curves({
    "dent": (rec, 1 - 0.35 * rec ** 6, 0.804),
    "hole": (rec, 1 - 0.45 * rec ** 4, 0.794),
    "rusty": (rec, 1 - 0.30 * rec ** 8, 0.833),
    "all": (rec, 1 - 0.38 * rec ** 5, 0.810),
}, ax=axes[1, 0])

# 4) 特征重要性（降序彩条）
names = ["新车价", "使用时长", "年款", "排量", "燃油类型_1", "里程", "品牌", "车身类型"]
vals = np.array([54.4, 9.5, 7.0, 3.7, 3.5, 2.9, 2.4, 1.8])
ps.plot_feature_importance(names, vals, ax=axes[1, 1])

fig.tight_layout()
out = Path.home() / "mathorcup" / "figures" / "demo_m_cup_style.png"
ps.save(fig, out)
print("saved", out)
