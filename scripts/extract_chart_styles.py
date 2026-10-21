#!/usr/bin/env python3
"""抽取 PDF 内嵌图片，识别图表并统计配色/背景/风格特征。"""
import io, re, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import fitz
from PIL import Image

REF = Path.home() / "mathorcup" / "docs" / "reference"
OUT = Path.home() / "mathorcup" / "docs" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# 常见工具默认色
PALETTES = {
    "matplotlib_tab10": ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"],
    "excel": ["#4472C4","#ED7D31","#A5A5A5","#FFC000","#5B9BD5","#70AD47"],
    "seaborn_deep": ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"],
    "origin": ["#000000","#FF0000","#0000FF","#008000","#FF00FF","#00FFFF"],
}
def hex2rgb(h): return tuple(int(h[i:i+2],16) for i in (1,3,5))
PAL_RGB = {k:[hex2rgb(c) for c in v] for k,v in PALETTES.items()}

def nearest_palette(color, thr=40):
    best = None; bd = 1e9
    for name, cols in PAL_RGB.items():
        for c in cols:
            d = sum((a-b)**2 for a,b in zip(color,c)) ** 0.5
            if d < bd: bd, best = d, name
    return best if bd <= thr else None

rows = []
pdfs = [p for p in REF.rglob("*.pdf") if re.search(r"优秀论文[-0-9]*\.pdf$", p.name)]
for pdf in pdfs:
    paper = pdf.stem
    try:
        doc = fitz.open(pdf)
    except Exception:
        continue
    n_chart = 0; hits = Counter(); whites = []; ncolors = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4: pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.width < 200 or pix.height < 150: continue
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:,:,:3]
            except Exception:
                continue
            h, w = arr.shape[:2]
            ar = w / h
            if not (0.3 <= ar <= 3.5): continue
            white = (arr > 240).all(axis=2).mean()
            if white < 0.45: continue          # 图表通常有大面积白底
            small = np.array(Image.fromarray(arr).resize((160, int(160*h/w))))
            pimg = Image.fromarray(small).quantize(colors=24)
            pal = pimg.getpalette()[:72]
            counts = np.array(pimg.histogram()[:24])
            idx = counts.argsort()[::-1]
            top = [(pal[i*3], pal[i*3+1], pal[i*3+2]) for i in idx[:10] if counts[i] > small.shape[0]*small.shape[1]*0.002]
            n_chart += 1
            whites.append(round(white, 2)); ncolors.append(len(top))
            for c in top:
                p = nearest_palette(tuple(c))
                if p: hits[p] += 1
    if n_chart:
        rows.append({"paper": paper, "charts": n_chart,
                     "palette_hits": dict(hits), "mean_white": round(float(np.mean(whites)),2),
                     "mean_colors": round(float(np.mean(ncolors)),1)})

json.dump(rows, open(OUT/"chart_styles.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("论文数:", len(rows))
agg = Counter()
for r in rows:
    for p, c in r["palette_hits"].items(): agg[p] += 1
print("== 配色工具命中论文数(含默认色即计) ==")
for k, v in agg.most_common(): print(k, v)
print("总图表图片数:", sum(r["charts"] for r in rows))
print("\n== 每篇图表数与主配色 ==")
for r in sorted(rows, key=lambda x: -x["charts"])[:15]:
    top = sorted(r["palette_hits"].items(), key=lambda x:-x[1])[:2]
    print(r["paper"][:28], "| 图表", r["charts"], "| 白底", r["mean_white"], "|", top)
