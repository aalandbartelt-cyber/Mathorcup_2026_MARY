#!/usr/bin/env python3
"""批量提取 PDF 文本，带页码标记与质量统计。"""
import sys
from pathlib import Path
import fitz  # PyMuPDF

REF = Path.home() / "mathorcup" / "docs" / "reference"
OUT = Path.home() / "mathorcup" / "docs" / "analysis" / "txt"
OUT.mkdir(parents=True, exist_ok=True)

pdfs = sorted(REF.rglob("*.pdf"))
print(f"共 {len(pdfs)} 份 PDF")
rows = []
for i, p in enumerate(pdfs, 1):
    rel = p.relative_to(REF)
    out = OUT / (str(rel).replace("/", "__").replace(".pdf", ".txt"))
    try:
        doc = fitz.open(p)
        parts, chars, empty = [], 0, 0
        for n, page in enumerate(doc, 1):
            t = page.get_text("text")
            if len(t.strip()) < 20:
                empty += 1
            chars += len(t)
            parts.append(f"\n===== PAGE {n} =====\n{t}")
        out.write_text("".join(parts), encoding="utf-8")
        rows.append((rel, len(doc), chars, empty))
        print(f"[{i}/{len(pdfs)}] {rel} | 页数 {len(doc)} | 字符 {chars} | 空白页 {empty}")
    except Exception as e:
        print(f"[{i}] 失败 {rel}: {e}")
print("\n== 汇总 ==")
total_pages = sum(r[1] for r in rows)
print(f"总页数 {total_pages} | 总字符 {sum(r[2] for r in rows)}")
sus = [r for r in rows if r[2] < 3000]
print(f"疑似扫描版(字符<3000): {len(sus)}")
for r in sus:
    print("  ", r[0], r[2])
