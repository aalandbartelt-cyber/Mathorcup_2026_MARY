#!/usr/bin/env python3
"""对提取的论文文本做结构统计:标题、章节、图表数、方法关键词。"""
import re, json
from pathlib import Path
from collections import Counter

TXT = Path.home() / "mathorcup" / "docs" / "analysis" / "txt"
OUT = Path.home() / "mathorcup" / "docs" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

KW = ["YOLO","R-CNN","CNN","ResNet","Transformer","注意力","LSTM","GRU","RNN","BERT","XGBoost","LightGBM",
      "随机森林","决策树","SVM","K-means","聚类","PCA","ARIMA","Prophet","灰色预测","时间序列","回归",
      "AHP","熵权","TOPSIS","模糊评价","主成分","因子分析","遗传算法","粒子群","模拟退火","蚁群",
      "整数规划","线性规划","动态规划","多目标","启发式","朴素贝叶斯","集成学习","Stacking","迁移学习",
      "数据增强","归一化","特征工程","交叉验证","网格搜索","贝叶斯优化","SHAP","可解释","灵敏度分析","鲁棒性"]

def find_title(lines):
    for l in lines[:40]:
        s = l.strip()
        if 6 <= len(s) <= 80 and not re.search(r"摘要|Abstract|第.届|编号|参赛队|学校", s):
            return s
    return ""

def sections(text):
    pats = [r"^\s*([一二三四五六七八九十]+)[、.．]\s*\S+", r"^\s*(第[一二三四五六七八九十\d]+[章节])\s*\S+",
            r"^\s*(\d+(?:\.\d+)?)\s+\S{2,20}$", r"^\s*(\d+(?:\.\d+){1,2})\s*\S+"]
    found = []
    for line in text.splitlines():
        s = line.strip()
        if 2 <= len(s) <= 40:
            for p in pats:
                if re.match(p, s):
                    found.append(s); break
    return found

rows = []
for f in sorted(TXT.glob("*.txt")):
    if "优秀论文" not in f.name:
        continue
    text = f.read_text(encoding="utf-8")
    pages = len(re.findall(r"===== PAGE", text))
    lines = text.splitlines()
    figs = len(re.findall(r"图\s*\d+", text)); tabs = len(re.findall(r"表\s*\d+", text))
    eqs = len(re.findall(r"[（(]\s*\d+\s*[)）]", text))
    kws = {k: text.count(k) for k in KW}
    kws = {k: v for k, v in kws.items() if v >= 3}
    secs = sections(text)
    rows.append({"file": f.name, "title": find_title(lines), "pages": pages, "chars": len(text),
                 "figs": figs, "tabs": tabs, "eqs": eqs, "kw": dict(sorted(kws.items(), key=lambda x: -x[1])[:8]),
                 "secs": secs[:14]})

json.dump(rows, open(OUT/"papers_stats.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"分析 {len(rows)} 篇\n")
for r in rows:
    short = r["file"].split("__")[-1][:26]
    kw_s = str(r["kw"])
    sec_join = " / ".join(r["secs"][:8])
    print("【" + short + "】" + r["title"][:50])
    print("  页 " + str(r["pages"]) + " | 图 " + str(r["figs"]) + " | 表 " + str(r["tabs"]) + " | 公式约 " + str(r["eqs"]) + " | 关键词: " + kw_s)
    print("  小节数 " + str(len(r["secs"])) + ": " + sec_join)
