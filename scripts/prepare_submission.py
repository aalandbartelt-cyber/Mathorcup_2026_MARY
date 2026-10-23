#!/usr/bin/env python3
"""按 2026 规范归档提交文件（改名为 题号+7位队号）。

用法：
  python scripts/prepare_submission.py --root . --team MCB2601605 --track A \
    --paper path/论文.pdf --result path/test_result.csv \
    --promise path/承诺书.pdf --support path/附件.zip

命名：MCB2601605 + A → A2601605.pdf / A2601605承诺书.pdf / A2601605附件.zip
"""
import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--team", required=True, help="报名队号，如 MCB2601605")
    ap.add_argument("--track", required=True, choices=["A", "B"])
    ap.add_argument("--paper", help="论文 PDF 路径")
    ap.add_argument("--result", help="结果文件或目录")
    ap.add_argument("--promise", help="签名后的承诺书（pdf/jpg/png）")
    ap.add_argument("--support", help="支撑材料压缩包（rar/zip）")
    args = ap.parse_args()

    team = args.team.strip().upper()
    numeric = team[3:] if team.startswith("MCB") else team
    if not numeric.isdigit() or len(numeric) != 7:
        raise SystemExit(f"队号解析失败：{team} → {numeric}（应为 MCB + 7 位数字）")
    base = f"{args.track}{numeric}"

    root = Path(args.root).expanduser().resolve()
    d_paper = root / "06_交付" / "01_论文"
    d_result = root / "06_交付" / "02_结果文件"
    d_promise = root / "06_交付" / "03_承诺书"
    d_support = root / "06_交付" / "04_支撑材料"
    for d in (d_paper, d_result, d_promise, d_support):
        d.mkdir(parents=True, exist_ok=True)

    print(f"队号 {team} → 文件基名 {base}")

    if args.paper:
        src = Path(args.paper).expanduser()
        if not src.exists():
            raise SystemExit(f"论文不存在：{src}")
        dst = d_paper / f"{base}.pdf"
        shutil.copy2(src, dst)
        print("论文  →", dst.relative_to(root))

    if args.result:
        src = Path(args.result).expanduser()
        if src.is_dir():
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, d_result / f.name)
                    print("结果  →", (d_result / f.name).relative_to(root))
        elif src.exists():
            shutil.copy2(src, d_result / src.name)
            print("结果  →", (d_result / src.name).relative_to(root))
        else:
            raise SystemExit(f"结果不存在：{src}")

    if args.promise:
        src = Path(args.promise).expanduser()
        if not src.exists():
            raise SystemExit(f"承诺书不存在：{src}")
        dst = d_promise / f"{base}承诺书{src.suffix.lower()}"
        shutil.copy2(src, dst)
        print("承诺书 →", dst.relative_to(root))

    if args.support:
        src = Path(args.support).expanduser()
        if not src.exists():
            raise SystemExit(f"支撑材料不存在：{src}")
        dst = d_support / f"{base}附件{src.suffix.lower()}"
        shutil.copy2(src, dst)
        print("支撑材料 →", dst.relative_to(root))

    print("\n下一步：")
    print(f"  python scripts/audit_mcup_submission.py --root . --paper 06_交付/01_论文/{base}.pdf "
          f"--result 06_交付/02_结果文件/<结果文件>")
    print("  按 06_交付/提交清单.md 逐项检查，再去赛氪提交")


if __name__ == "__main__":
    main()
