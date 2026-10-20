#!/usr/bin/env python3
"""通用 train/val 分层划分(YOLO 格式)。

按"图像包含的类别组合"分层，保证稀有类别在验证集中的比例不失控。

用法:
    python scripts/split_train_val.py --images data/<数据集>/images/train \
                                      --labels data/<数据集>/labels/train \
                                      --val-ratio 0.2 --seed 42

说明: 默认用 os.rename 移动文件(同盘瞬间完成)，验证集目录自动创建。
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="训练图片目录(划分后仍放这里)")
    ap.add_argument("--labels", required=True, help="训练标签目录")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-images", default=None, help="验证图片目录(默认 images/val)")
    ap.add_argument("--val-labels", default=None, help="验证标签目录(默认 labels/val)")
    args = ap.parse_args()

    img_dir, lbl_dir = Path(args.images), Path(args.labels)
    out_img = Path(args.val_images) if args.val_images else img_dir.parent / "val"
    out_lbl = Path(args.val_labels) if args.val_labels else lbl_dir.parent / "val"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    groups: dict[tuple, list] = defaultdict(list)
    for f in files:
        lbl = lbl_dir / (f.stem + ".txt")
        classes = ()
        if lbl.exists():
            classes = tuple(sorted({int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}))
        groups[classes].append(f)

    moved = 0
    for cls, fs in groups.items():
        random.shuffle(fs)
        n_val = max(1, round(len(fs) * args.val_ratio))
        for f in fs[:n_val]:
            f.rename(out_img / f.name)
            lbl = lbl_dir / (f.stem + ".txt")
            if lbl.exists():
                lbl.rename(out_lbl / lbl.name)
            moved += 1

    print(f"验证集: {moved} 张 | 训练集剩余: {len(files) - moved} 张")
    for cls in sorted(groups):
        print(f"  类别组 {cls}: 总 {len(groups[cls])} 张, 划出 {max(1, round(len(groups[cls]) * args.val_ratio))} 张")


if __name__ == "__main__":
    main()
