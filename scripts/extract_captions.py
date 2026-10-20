#!/usr/bin/env python3
"""统计全部论文的图/表标题类型分布。"""
import re
from collections import Counter
from pathlib import Path

TXT = Path.home() / "mathorcup" / "docs" / "analysis" / "txt"
OUT = Path.home() / "mathorcup" / "docs" / "analysis" / "captions.md"

fig_pat = re.compile(r"图\s*\d+[-–—.]?\d*\s*([^\n]{2,60})")
tab_pat = re.compile(r"表\s*\d+[-–—.]?\d*\s*([^\n]{2,60})")

CATS = {
    "流程图/技术路线": ["流程", "技术路线", "框架", "总体", "思路", "步骤"],
    "网络结构图": ["结构图", "网络结构", "主干", "模块", "注意力机制图", "backbone", "架构"],
    "数据分布图": ["分布", "直方图", "密度", "KDE", "箱线", "小提琴"],
    "相关性图": ["相关", "热力图", "散点", "矩阵"],
    "训练过程图": ["损失", "loss", "收敛", "训练曲线", "精度变化", "迭代"],
    "混淆矩阵": ["混淆矩阵", "confusion"],
    "PR/ROC曲线": ["PR", "P-R", "ROC", "AUC", "精确率-召回率"],
    "检测结果图": ["检测结果", "识别结果", "预测框", "可视化", "对比图", "示例"],
    "柱状/条形图": ["柱状", "条形", "bar"],
    "折线图": ["折线", "趋势"],
    "饼图": ["饼图"],
    "词云": ["词云"],
    "地图/空间": ["地图", "空间分布", "轨迹", "路径"],
    "消融/对比": ["消融", "对比实验", "不同模型", "比较"],
    "评价指标表": ["评估指标", "评价指标", "指标结果", "性能对比", "误差"],
}

fig_counter, tab_counter, all_figs, all_tabs = Counter(), Counter(), [], []
for f in sorted(TXT.glob("*优秀论文*.txt")):
    if "优秀论文" not in f.name: continue
    t = f.read_text(encoding="utf-8")
    name = f.name.split("__")[-1].replace(".txt","")
    for m in fig_pat.finditer(t):
        cap = m.group(1).strip()
        if len(cap) < 3: continue
        all_figs.append((name, cap))
        for cat, kws in CATS.items():
            if any(k.lower() in cap.lower() for k in kws):
                fig_counter[cat] += 1; break
    for m in tab_pat.finditer(t):
        cap = m.group(1).strip()
        if len(cap) < 3: continue
        all_tabs.append((name, cap))

print("== 图类型分布(按标题关键词) ==")
for k, v in fig_counter.most_common():
    print(f"{k}: {v}")
print(f"总图标题捕获数: {len(all_figs)} | 总表标题捕获数: {len(all_tabs)}")

# 抽样: 每类给出3个真实标题示例
samples = {}
for name, cap in all_figs:
    for cat, kws in CATS.items():
        if any(k.lower() in cap.lower() for k in kws):
            samples.setdefault(cat, []).append((name, cap)); break

lines = ["# 图表标题语料(全部论文)\n", "## 类型统计\n"]
for k, v in fig_counter.most_common():
    lines.append(f"- {k}: {v}")
lines.append("\n## 真实标题示例\n")
for cat, items in samples.items():
    lines.append(f"### {cat}")
    for name, cap in items[:6]:
        lines.append(f"- 「{cap}」 — {name}")
    lines.append("")
lines.append("\n## 全部图标题\n")
for name, cap in all_figs:
    lines.append(f"- {name}: {cap}")
lines.append("\n## 全部表标题\n")
for name, cap in all_tabs:
    lines.append(f"- {name}: {cap}")
OUT.write_text("\n".join(lines), encoding="utf-8")
print("写出", OUT, f"{OUT.stat().st_size/1024:.0f} KB")
