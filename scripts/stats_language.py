#!/usr/bin/env python3
"""统计论文的章节命名习惯、摘要句式、指标词频。"""
import re
from collections import Counter
from pathlib import Path

TXT = Path.home() / "mathorcup" / "docs" / "analysis" / "txt"
papers = [f for f in sorted(TXT.glob("*优秀论文*.txt")) if "优秀论文" in f.name]
print(f"论文数: {len(papers)}")

heads = {
    "摘要": r"摘\s*要", "关键词": r"关键词", "目录": r"目\s*录",
    "问题重述": r"问题重述|问题的重述", "问题背景": r"问题背景|研究背景",
    "问题分析": r"问题分析|问题的分析", "模型假设": r"模型假设|模型的假设",
    "符号说明": r"符号说明|符号定义", "数据预处理": r"数据预处理|数据清洗|数据探索",
    "模型建立与求解": r"模型的建立与求解|模型建立与求解|建模与求解",
    "结果分析": r"结果分析|求解与分析|实验结果", "模型检验": r"模型检验|灵敏度分析|稳健性",
    "模型评价": r"模型评价|模型的评价", "模型推广": r"模型推广|模型的推广|改进与推广",
    "总结与展望": r"总结与展望|研究总结", "参考文献": r"参考文献", "附录": r"附\s*录",
}
hcount = Counter()
for f in papers:
    t = f.read_text(encoding="utf-8")
    for name, pat in heads.items():
        if re.search(pat, t): hcount[name] += 1
print("\n== 章节出现率 ==")
for k, v in hcount.most_common():
    print(f"{k}: {v}/{len(papers)} ({v/len(papers)*100:.0f}%)")

metrics = ["Accuracy","准确率","Precision","精确率","Recall","召回率","F1","mAP","IoU","Dice",
           "AUC","ROC","R2","R²","MAE","RMSE","MAPE","wmape","WMAE","MSE","AIC","轮廓系数",
           "灵敏度","鲁棒性","泛化","消融","交叉验证","网格搜索","贝叶斯优化","SHAP","Optuna"]
print("\n== 指标/方法词频(出现论文数) ==")
mc = Counter()
for f in papers:
    t = f.read_text(encoding="utf-8")
    for m in metrics:
        if t.count(m) >= 2: mc[m] += 1
for k, v in mc.most_common():
    print(f"{k}: {v} 篇")

phr = ["针对问题一","针对问题二","针对问题三","针对问题1","针对问题2","针对问题3",
       "本文","首先","其次","最后","结果表明","实验结果表明","综上所述","显著","优于",
       "提升","验证了","达到了","分别为","如表","如图","见图","算法流程","技术路线"]
print("\n== 摘要/正文高频表述(总次数) ==")
pc = Counter()
for f in papers:
    t = f.read_text(encoding="utf-8")
    for p in phr: pc[p] += t.count(p)
for k, v in pc.most_common():
    print(f"{k}: {v}")
