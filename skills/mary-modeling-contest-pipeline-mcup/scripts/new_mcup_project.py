#!/usr/bin/env python3
"""建立 MathorCup 大数据竞赛项目骨架。

用法：
  python scripts/new_mcup_project.py --root "D:/mm/mcup2026" --track A
  python scripts/new_mcup_project.py --root ~/mcup --track B --questions 3
"""
import argparse
import shutil
from pathlib import Path

DIRS = [
    "00_赛题与规则", "01_数据", "02_代码", "03_结果", "04_图表",
    "05_论文", "06_交付", "docs/collab", "_archive",
]

PARAM_TMPL = """# 本届赛事参数（开赛 10 分钟内填完）

P1 赛事与届次：        MathorCup 大数据竞赛（2026 第七届）
P2 赛道：              {track}（A=视觉 / B=表格；只能选一个）
P3 作品语言/页数：      中文；正文 ≤30 页（附录不限）
P4 电子版第一页：       题目 + 摘要 + 关键词；第二页目录；正文页码从 1
P5 附录与代码：         须含软件名称、命令、全部源程序
P6 支撑材料：           可选压缩包；结果文件按赛题要求
P7 匿名：               全文 + 元数据 + 代码路径不得出现学校/姓名/队号
（附加）提交：          论文 PDF（#****.pdf）+ 结果文件 + 承诺书
（附加）开赛：          2026-10-23 18:00（以官网当届通知为准）
"""

TOPIC_TMPL = """# 选题结论

## 已知
- 数据：
- 字段/单位：
- 约束：

## 要求（逐字抄题面）
- 问题一：
- 问题二：
- 问题三：

## 隐含
-

## 陷阱
- 数据泄漏：
- 类别不平衡：
- 坐标/单位口径：
- 划分方式：
"""

AUDIT_TMPL = """# 数据对账表

| 项目 | 题面 | 实测 | 差异处理 |
|---|---|---|---|
| 文件数 |  |  |  |
| 图像尺寸/模式 |  |  |  |
| 字段/单位 |  |  |  |
| 类别分布 |  |  |  |
| 缺失率 |  |  |  |
| 划分方式 |  |  |  |
| 泄漏风险 |  |  |  |
"""

README = """# {name}

MathorCup 大数据竞赛项目（{track} 赛道）。

## 快速开始
1. 填 `00_赛题与规则/赛事参数.md` 与 `选题结论.md`
2. 数据放 `01_数据/`，完成 `数据对账表.md`
3. 结果模板：`03_结果/templates/`
4. 出图：`Rscript -e 'source("scripts/plot_style.R")'`（或复制技能 assets/r_theme_mcup.R）
5. 提交前：`python scripts/audit_mcup_submission.py --root . --paper 06_交付/论文.pdf --result 03_结果/result.csv`
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--track", choices=["A", "B"], default="A")
    ap.add_argument("--questions", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)

    (root / "00_赛题与规则" / "赛事参数.md").write_text(
        PARAM_TMPL.format(track=args.track), encoding="utf-8")
    (root / "00_赛题与规则" / "选题结论.md").write_text(TOPIC_TMPL, encoding="utf-8")
    (root / "01_数据" / "数据对账表.md").write_text(AUDIT_TMPL, encoding="utf-8")
    (root / "README.md").write_text(
        README.format(name=root.name, track=args.track), encoding="utf-8")

    # 结果模板与实验记录
    skill = Path(__file__).resolve().parents[1]
    tpl = skill / "assets" / "result_templates"
    if tpl.exists():
        dst = root / "03_结果" / "templates"
        dst.mkdir(parents=True, exist_ok=True)
        for f in tpl.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        log = tpl / "experiment_log.md"
        if log.exists():
            shutil.copy2(log, root / "docs" / "experiment_log.md")

    print(f"已建立：{root}")
    for d in DIRS:
        print("  ", d)
    print("\n下一步：填赛事参数 → 数据对账 → 基线 → 论文骨架")
    print("论文骨架见技能 assets/paper_outline.md；检查清单见 assets/checklists.md")


if __name__ == "__main__":
    main()
