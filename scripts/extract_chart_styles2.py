#!/usr/bin/env python3
"""更严格的图表配色识别：精确色距 + 同图多色判别 + 灰度检测。"""
import re
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image

REF = Path.home() / "mathorcup" / "docs" / "reference"
OUT = Path.home() / "mathorcup" / "docs" / "analysis"

PALETTES = {
    "mpl_tab10": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "excel": ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5", "#70AD47"],
    "sns_deep": ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                 "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    "origin": ["#000000", "#FF0000", "#0000FF", "#008000", "#FF00FF"],
}


def h2r(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])


PAL = {k: [h2r(c) for c in v] for k, v in PALETTES.items()}

rows = []
pdfs = [p for p in REF.rglob("*.pdf") if re.search(r"优秀论文[-0-9]*\.pdf$", p.name)]
for pdf in pdfs:
    paper = pdf.stem
    try:
        doc = fitz.open(pdf)
    except Exception:
        continue
    charts = 0
    pal_exact = Counter()
    grayscale = 0
    whites, ars = [], []
    for page in doc:
        for img in page.get_images(full=True):
            try:
                pix = fitz.Pixmap(doc, img[0])
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.width < 260 or pix.height < 180:
                    continue
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)[:, :, :3]
            except Exception:
                continue
            h, w = arr.shape[:2]
            ar = w / h
            if not (0.35 <= ar <= 3.0):
                continue
            white = (arr > 240).all(axis=2).mean()
            if white < 0.5:
                continue
            charts += 1
            whites.append(white)
            ars.append(ar)
            sm = np.array(Image.fromarray(arr).resize((200, int(200 / ar))))
            q = Image.fromarray(sm).quantize(colors=24)
            pal = np.array(q.getpalette()[:72]).reshape(-1, 3)
            cnt = np.array(q.histogram()[:24])
            total = sm.shape[0] * sm.shape[1]
            keep = [pal[i] for i in range(24)
                    if cnt[i] >= total * 0.015 and not (pal[i] > 235).all()]
            if keep:
                sat = np.mean([int(c.max()) - int(c.min()) for c in keep])
                if sat < 12:
                    grayscale += 1
            hits = set()
            for c in keep:
                for name, cols in PAL.items():
                    if min(np.linalg.norm(c - x) for x in cols) <= 10:
                        hits.add((name, tuple(int(v) for v in c)))
            for name, _ in hits:
                pal_exact[name] += 1
    if charts:
        rows.append(dict(paper=paper, charts=charts, palette=dict(pal_exact),
                         grayscale=grayscale,
                         white=round(float(np.mean(whites)), 2),
                         ar=round(float(np.mean(ars)), 2)))

json.dump(rows, open(OUT / "chart_styles.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("论文数", len(rows), "图表数", sum(r["charts"] for r in rows))

agg = Counter()
gray_papers = 0
for r in rows:
    for k, v in r["palette"].items():
        if v >= 2:
            agg[k] += 1
    if r["grayscale"] >= max(2, 0.3 * r["charts"]):
        gray_papers += 1
print("== 至少2个精确命中颜色的论文数 ==")
for k, v in agg.most_common():
    print(k, v)
print("灰度图为主(>=30%图表)的论文数:", gray_papers)
print()
print("== 每篇签名(图表数 | 白底 | 宽高比 | 配色命中 | 灰度图数) ==")
for r in sorted(rows, key=lambda x: -x["charts"]):
    name = r["paper"][:26]
    print(name.ljust(28), str(r["charts"]).rjust(3), "|", r["white"], "|",
          r["ar"], "|", dict(r["palette"]), "| 灰" + str(r["grayscale"]))
