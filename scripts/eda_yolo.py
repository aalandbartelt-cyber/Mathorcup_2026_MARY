#!/usr/bin/env python3
"""YOLO 格式数据集的通用 EDA。

用法:
    python scripts/eda_yolo.py --images data/<数据集>/images/train \
                               --labels data/<数据集>/labels/train \
                               --classes data/<数据集>/classes.txt \
                               --out figures/eda

输出 4 张论文级图 + 终端摘要:
  1) 各类别标注框数量
  2) 每张图的标注框数量分布
  3) 标注框面积分布(对数坐标，分析多尺度)
  4) 标注框宽高散点图
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图片目录")
    ap.add_argument("--labels", required=True, help="YOLO 标签目录")
    ap.add_argument("--classes", default=None, help="classes.txt(可选)")
    ap.add_argument("--out", default="figures/eda", help="输出目录")
    args = ap.parse_args()

    img_dir, lbl_dir = Path(args.images), Path(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    label_files = list(lbl_dir.glob("*.txt"))

    names: dict[int, str] = {}
    if args.classes and Path(args.classes).exists():
        names = {i: n.strip() for i, n in enumerate(Path(args.classes).read_text(encoding="utf-8").splitlines()) if n.strip()}

    class_counts: Counter = Counter()
    per_image: list[int] = []
    widths, heights, areas = [], [], []
    missing = empty = 0

    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing += 1
            per_image.append(0)
            continue
        lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        if not lines:
            empty += 1
            per_image.append(0)
            continue
        per_image.append(len(lines))
        for l in lines:
            c = int(l[0])
            w, h = float(l[3]), float(l[4])
            class_counts[c] += 1
            widths.append(w)
            heights.append(h)
            areas.append(w * h)

    total_boxes = int(sum(class_counts.values()))
    print(f"图片数: {len(images)} | 标签文件: {len(label_files)}")
    print(f"缺标签的图: {missing} | 空标签(无目标)的图: {empty} | 总框数: {total_boxes}")
    for c in sorted(class_counts):
        print(f"  类别 {c} ({names.get(c, '?')}): {class_counts[c]} 框")
    if areas:
        a = np.array(areas) * 640 * 640
        print(f"框面积(按640px计, px^2) 分位数 P10/P50/P90: {np.percentile(a, 10):.0f} / {np.percentile(a, 50):.0f} / {np.percentile(a, 90):.0f}")
        print(f"小框(<32^2)占比: {(a < 1024).mean() * 100:.1f}% | 大框(>96^2)占比: {(a > 9216).mean() * 100:.1f}%")
    print(f"每图框数: 均值 {np.mean(per_image):.2f} 最大 {max(per_image)}")

    # 1) 类别分布
    fig, ax = plt.subplots(figsize=(6, 4))
    cs = sorted(class_counts)
    ax.bar([names.get(c, str(c)) for c in cs], [class_counts[c] for c in cs], color="#4C72B0")
    ax.set_ylabel("标注框数量")
    ax.set_title("各类别标注框数量分布")
    fig.savefig(out_dir / "class_distribution.png")
    plt.close(fig)

    # 2) 每图框数
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_image, bins=range(0, max(per_image) + 2), color="#55A868", edgecolor="white")
    ax.set_xlabel("每张图的标注框数量")
    ax.set_ylabel("图片数量")
    ax.set_title("每图标注框数量分布")
    fig.savefig(out_dir / "boxes_per_image.png")
    plt.close(fig)

    # 3) 框面积分布(对数)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(areas, bins=60, color="#C44E52", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("归一化框面积(对数)")
    ax.set_ylabel("数量")
    ax.set_title("标注框面积分布(多尺度分析)")
    fig.savefig(out_dir / "box_area_distribution.png")
    plt.close(fig)

    # 4) 宽高散点
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(widths, heights, s=6, alpha=0.35, color="#8172B2")
    ax.set_xlabel("归一化宽")
    ax.set_ylabel("归一化高")
    ax.set_title("标注框宽高分布")
    fig.savefig(out_dir / "box_wh_scatter.png")
    plt.close(fig)

    print(f"图表已保存到 {out_dir.resolve()}")


if __name__ == "__main__":
    main()
