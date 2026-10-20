#!/usr/bin/env python3
"""为每篇论文提取:摘要、问题重述开头、三个问题的描述、结论段,输出汇总文件。"""
import re
from pathlib import Path

TXT = Path.home() / "mathorcup" / "docs" / "analysis" / "txt"
OUT = Path.home() / "mathorcup" / "docs" / "analysis" / "contexts.md"

def clean(t):
    t = re.sub(r"===== PAGE \d+ =====", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def seg(text, start_pat, max_chars=900):
    m = re.search(start_pat, text)
    if not m:
        return ""
    return text[m.start():m.start()+max_chars].replace("\n", " ")

parts = []
for f in sorted(TXT.glob("*优秀论文*.txt")):
    name = f.name.split("__")[-1].replace(".txt","")
    t = clean(f.read_text(encoding="utf-8"))
    abstract = seg(t, r"摘\s*要.{0,10}", 700)
    restate  = seg(t, r"(问题重述|问题的重述|问题背景)", 500)
    p1 = seg(t, r"(针对问题一|问题一的分析|问题1\s*的分析|问题1的?分析)", 260)
    p2 = seg(t, r"(针对问题二|问题二的分析|问题2\s*的分析|问题2的?分析)", 260)
    p3 = seg(t, r"(针对问题三|问题三的分析|问题3\s*的分析|问题3的?分析)", 260)
    concl = seg(t, r"(模型评价|模型的评价|模型推广|优缺点)", 400)
    parts.append(f"""## {name}

**摘要节选**: {abstract}

**问题重述**: {restate}

**问题一分析**: {p1}

**问题二分析**: {p2}

**问题三分析**: {p3}

**模型评价**: {concl}
""")

OUT.write_text("\n".join(parts), encoding="utf-8")
print(f"写入 {OUT} | 大小 {OUT.stat().st_size/1024:.0f} KB")
