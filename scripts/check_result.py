#!/usr/bin/env python3
"""结果文件校验：分类 CSV / 检测 CSV / 分配 TXT。

用法：
  python scripts/check_result.py --file outputs/result_detection.csv --images-dir data/test/images
  python scripts/check_result.py --file outputs/result_classification.csv
  python scripts/check_result.py --file outputs/result_assignment.txt

校验内容：表头、数值范围（检测坐标应为 0–1 归一化）、类别 id、重复行、
图片覆盖率、缺图/多图警告。退出码 1 表示存在错误。
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise SystemExit("文件为空")
    return rows[0], rows[1:]


def is_float(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def check_classification(header, rows):
    errs, warns = [], []
    need = ["image_id", "prediction"]
    for c in need:
        if c not in header:
            errs.append(f"缺少字段 {c}（当前表头：{header}）")
    if errs:
        return errs, warns, {}
    idx = {c: header.index(c) for c in header}
    ids = Counter()
    labels = Counter()
    for i, r in enumerate(rows, 2):
        if len(r) < len(header):
            errs.append(f"第 {i} 行字段数不足：{r}")
            continue
        image_id = r[idx["image_id"]].strip()
        if not image_id:
            errs.append(f"第 {i} 行 image_id 为空")
        ids[image_id] += 1
        lab = r[idx["prediction"]].strip()
        if lab == "":
            warns.append(f"第 {i} 行 prediction 为空（若代表无目标可忽略）")
        labels[lab] += 1
        if "probability" in idx and r[idx["probability"]].strip():
            p = r[idx["probability"]].strip()
            if not is_float(p) or not (0 <= float(p) <= 1):
                errs.append(f"第 {i} 行 probability 不在 [0,1]：{p}")
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        warns.append(f"{len(dup)} 个 image_id 出现多次（分类任务通常应唯一）")
    return errs, warns, {"rows": len(rows), "images": len(ids), "labels": dict(labels)}


def check_detection(header, rows, images_dir):
    errs, warns = [], []
    need = ["image_id", "class_id", "x_center", "y_center", "width", "height"]
    for c in need:
        if c not in header:
            errs.append(f"缺少字段 {c}（当前表头：{header}）")
    if errs:
        return errs, warns, {}
    idx = {c: header.index(c) for c in header}
    ids = Counter()
    cls = Counter()
    for i, r in enumerate(rows, 2):
        if len(r) < len(header):
            errs.append(f"第 {i} 行字段数不足：{r}")
            continue
        image_id = r[idx["image_id"]].strip()
        ids[image_id] += 1
        try:
            c = int(float(r[idx["class_id"]]))
            x = float(r[idx["x_center"]])
            y = float(r[idx["y_center"]])
            w = float(r[idx["width"]])
            h = float(r[idx["height"]])
        except Exception:
            errs.append(f"第 {i} 行存在非数值：{r}")
            continue
        cls[c] += 1
        if not (0 <= x <= 1 and 0 <= y <= 1):
            errs.append(f"第 {i} 行中心点不在 [0,1]：x={x}, y={y}（是否误用像素坐标？）")
        if not (0 < w <= 1 and 0 < h <= 1):
            errs.append(f"第 {i} 行宽高不在 (0,1]：w={w}, h={h}")
        eps = 1e-6
        if x - w / 2 < -eps or x + w / 2 > 1 + eps or y - h / 2 < -eps or y + h / 2 > 1 + eps:
            warns.append(f"第 {i} 行框超出图像边界：({x},{y},{w},{h})")
    if images_dir:
        d = Path(images_dir)
        files = {p.name for p in d.iterdir() if p.suffix.lower() in EXTS} if d.is_dir() else set()
        if files:
            missing = [f for f in files if f not in ids]
            extra = [k for k in ids if k not in files]
            if missing:
                warns.append(f"{len(missing)} 张测试图没有预测行（如要求每图至少一行，需检查）")
            if extra:
                warns.append(f"{len(extra)} 个预测行对应的图片不在测试目录（示例：{extra[:3]}）")
    return errs, warns, {"rows": len(rows), "images": len(ids), "classes": dict(cls)}


def check_assignment(header_or_rows, rows, batch=False):
    """分配 TXT：官方无表头，逐行校验列数与整数性。"""
    errs, warns = [], []
    ncol = 5 if batch else 3
    for i, r in enumerate(rows, 1):
        r = [x.strip() for x in r]
        if not r or (r and r[0].startswith("#")):
            continue
        if len(r) != ncol:
            errs.append(f"第 {i} 行列数为 {len(r)}，应为 {ncol}：{r}")
            continue
        for v in r:
            try:
                int(float(v))
            except Exception:
                errs.append(f"第 {i} 行存在非整数：{r}")
                break
        if batch and len(r) == 5:
            if int(float(r[4])) not in (0, 1):
                errs.append(f"第 {i} 行 retainable 应为 0/1：{r}")
            if int(float(r[3])) == -1 and int(float(r[2])) != -1:
                warns.append(f"第 {i} 行 auntId=-1 但 serviceStartTime 不为 -1（压单口径需核对）")
    return errs, warns, {"rows": len([r for r in rows if r and not r[0].strip().startswith('#')])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--format", choices=["auto", "cls", "det", "assign", "assign_batch"], default="auto")
    args = ap.parse_args()
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"文件不存在：{path}")

    fmt = args.format
    if fmt == "auto":
        with open(path, encoding="utf-8-sig") as f:
            head = f.readline().strip()
        if "x_center" in head or "class_id" in head:
            fmt = "det"
        elif "prediction" in head:
            fmt = "cls"
        elif path.suffix.lower() in {".txt", ".csv"}:
            fmt = "assign"
        else:
            raise SystemExit("无法自动识别格式，请用 --format 指定")

    if fmt == "assign" or fmt == "assign_batch":
        with open(path, encoding="utf-8-sig") as f:
            rows = [line.strip().split(",") for line in f if line.strip()]
        errs, warns, stats = check_assignment(None, rows, batch=(fmt == "assign_batch"))
        title = "分配结果（分批）" if fmt == "assign_batch" else "分配结果"
    else:
        header, rows = read_csv(path)
        title = "检测结果" if fmt == "det" else "分类结果"
        if fmt == "det":
            errs, warns, stats = check_detection(header, rows, args.images_dir)
        else:
            errs, warns, stats = check_classification(header, rows)

    print(f"== 校验：{title} | {path}")
    print("统计:", stats)
    if ok := (not errs):
        print("✅ 无错误")
    for e in errs:
        print("❌", e)
    for w in warns:
        print("⚠️ ", w)
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
