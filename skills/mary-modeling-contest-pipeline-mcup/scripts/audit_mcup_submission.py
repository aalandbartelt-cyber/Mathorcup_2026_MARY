#!/usr/bin/env python3
"""MathorCup 提交前审计：论文页数/元数据、结果文件格式、必交文件。

用法：
  python scripts/audit_mcup_submission.py --root . --paper 06_交付/论文.pdf \
      --result 03_结果/result_detection.csv --images-dir 01_数据/test
"""
import argparse
import csv
import re
import sys
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_paper(path: Path, max_pages: int):
    errs, warns = [], []
    if not path.exists():
        return [f"论文不存在：{path}"], warns, {}
    if not re.match(r"^[AB]\d+\.pdf$", path.name):
        warns.append(f"文件名不符合 #****.pdf 规范：{path.name}")
    info = {"file": path.name, "size_mb": round(path.stat().st_size / 1048576, 2)}
    try:
        import pymupdf as fitz  # noqa: WPS433
        doc = fitz.open(path)
        info["pages"] = len(doc)
        if len(doc) > max_pages:
            errs.append(f"总页数 {len(doc)} 超过上限 {max_pages}（正文页数需另核）")
        meta = doc.metadata or {}
        for k in ("author", "creator", "title", "subject", "keywords"):
            if (meta.get(k) or "").strip():
                errs.append(f"PDF 元数据 {k} 非空：{meta.get(k)!r}（匿名违规风险）")
        doc.close()
    except ImportError:
        warns.append("未安装 pymupdf，跳过页数与元数据检查")
    except Exception as e:
        warns.append(f"PDF 解析失败：{e}")
    return errs, warns, info


def check_result(path: Path, images_dir: Path | None):
    errs, warns = [], []
    if not path.exists():
        return [f"结果文件不存在：{path}"], warns, {}
    if path.suffix.lower() == ".txt":
        rows = [r.strip().split(",") for r in path.read_text(encoding="utf-8").splitlines()
                if r.strip() and not r.strip().startswith("#")]
        for i, r in enumerate(rows, 1):
            if len(r) not in (3, 5):
                errs.append(f"第 {i} 行列数 {len(r)}（应为 3 或 5）：{r}")
        return errs, warns, {"rows": len(rows), "format": "assignment"}

    with open(path, encoding="utf-8-sig", newline="") as f:
        data = list(csv.reader(f))
    if not data:
        return ["结果 CSV 为空"], warns, {}
    header, rows = data[0], data[1:]
    info = {"format": "unknown", "rows": len(rows)}

    if "x_center" in header or "class_id" in header:
        need = ["image_id", "class_id", "x_center", "y_center", "width", "height"]
        missing = [c for c in need if c not in header]
        if missing:
            errs.append(f"缺少字段：{missing}")
        idx = {c: header.index(c) for c in header if c in need}
        ids = set()
        for i, r in enumerate(rows, 2):
            try:
                x, y = float(r[idx["x_center"]]), float(r[idx["y_center"]])
                w, h = float(r[idx["width"]]), float(r[idx["height"]])
                int(float(r[idx["class_id"]]))
            except Exception:
                errs.append(f"第 {i} 行存在非数值")
                continue
            ids.add(r[idx["image_id"]])
            if not (0 <= x <= 1 and 0 <= y <= 1):
                errs.append(f"第 {i} 行中心点不在 [0,1]（像素坐标？）")
            if not (0 < w <= 1 and 0 < h <= 1):
                errs.append(f"第 {i} 行宽高不在 (0,1]")
        info.update({"format": "detection", "images": len(ids)})
    elif "prediction" in header:
        idx = {c: header.index(c) for c in header}
        ids = set()
        for i, r in enumerate(rows, 2):
            ids.add(r[idx["image_id"]])
            if "probability" in idx and r[idx["probability"]].strip():
                p = float(r[idx["probability"]])
                if not (0 <= p <= 1):
                    errs.append(f"第 {i} 行 probability 不在 [0,1]：{p}")
        info.update({"format": "classification", "images": len(ids)})
    else:
        errs.append(f"无法识别的表头：{header}")

    if images_dir and images_dir.is_dir():
        files = {p.name for p in images_dir.iterdir() if p.suffix.lower() in EXTS}
        missing = files - ids
        extra = ids - files
        if missing:
            warns.append(f"{len(missing)} 张测试图无预测行")
        if extra:
            warns.append(f"{len(extra)} 个预测行对应图片不在测试目录")
        info["test_images"] = len(files)
    return errs, warns, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--paper", default="06_交付/论文.pdf")
    ap.add_argument("--result", default="03_结果/result.csv")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--max-pages", type=int, default=30)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    paper = (root / args.paper) if not Path(args.paper).is_absolute() else Path(args.paper)
    result = (root / args.result) if not Path(args.result).is_absolute() else Path(args.result)
    images = Path(args.images_dir) if args.images_dir else None

    e1, w1, i1 = check_paper(paper, args.max_pages)
    e2, w2, i2 = check_result(result, images)
    errs, warns = e1 + e2, w1 + w2

    print("== 论文 ==", i1)
    print("== 结果 ==", i2)
    for w in warns:
        print("⚠️ ", w)
    for e in errs:
        print("❌", e)
    if not errs:
        print("✅ 无硬性错误（仍需人工核对正文页数、赛道、承诺书）")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
