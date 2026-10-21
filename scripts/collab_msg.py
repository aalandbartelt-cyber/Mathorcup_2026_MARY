#!/usr/bin/env python3
"""ZCode ↔ Codex 文件协作：追加消息 / 更新状态。

用法：
  # 追加消息
  python scripts/collab_msg.py --to codex --kind task --title "跑通练习2推理" \
      --body "按 docs/2026练习题精解与赛题技术映射.md §3 执行" \
      --accept "输出 outputs/ct_pred.csv 且 check_result.py 通过" \
      --related "docs/2026练习题精解与赛题技术映射.md"

  # 改状态（接收方执行）
  python scripts/collab_msg.py --mark MSG-20261021-1030-a1 --status DONE --note "产物在 outputs/ct_pred.csv"

说明：
  --to codex  写入 docs/collab/to_codex.md（发起人默认 ZCode）
  --to zcode  写入 docs/collab/to_zcode.md（发起人默认 Codex）
  状态：OPEN / ACK / DONE / BLOCKED
"""
import argparse
import secrets
import sys
from datetime import datetime
from pathlib import Path


def default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now() -> datetime:
    return datetime.now()


def log_activity(root: Path, line: str) -> None:
    p = root / "docs" / "collab" / "board.md"
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    p.write_text(text + f"- [{now():%Y-%m-%d %H:%M}] {line}\n", encoding="utf-8")


def append_msg(root: Path, to: str, kind: str, title: str, body: str,
               accept: str, related: str, owner: str | None) -> str:
    fname = "to_codex.md" if to == "codex" else "to_zcode.md"
    path = root / "docs" / "collab" / fname
    ts = now()
    mid = f"MSG-{ts:%Y%m%d-%H%M}-{secrets.token_hex(1)}"
    sender = owner or ("ZCode" if to == "codex" else "Codex")
    receiver = "Codex" if to == "codex" else "ZCode"
    block = (
        f"\n## {mid} | {kind} | OPEN\n"
        f"- 发起: {sender}\n"
        f"- 接收: {receiver}\n"
        f"- 时间: {ts:%Y-%m-%d %H:%M}\n"
        f"- 关联: {related or '—'}\n\n"
        f"**标题**: {title}\n\n"
        f"**内容**:\n{body or '（略）'}\n\n"
        f"**验收**:\n{accept or '（略）'}\n\n"
        f"**回复**: （由 {receiver} 填写）\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(block)
    log_activity(root, f"{sender}: 发出 {mid}（{kind}）→ {receiver}：{title}")
    print(mid)
    return mid


def mark(root: Path, mid: str, status: str, note: str | None) -> None:
    for fname in ("to_codex.md", "to_zcode.md"):
        p = root / "docs" / "collab" / fname
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        if mid not in text:
            continue
        lines = text.splitlines()
        start = next((i for i, l in enumerate(lines) if l.startswith(f"## {mid} ")), None)
        if start is None:
            continue
        parts = lines[start].split("|")
        if len(parts) >= 3:
            parts[-1] = f" {status}"
            lines[start] = "|".join(parts)
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## MSG-")), len(lines))
        actor = "ZCode" if fname == "to_codex.md" else "Codex"
        ts = now().strftime("%Y-%m-%d %H:%M")
        suffix = f"：{note}" if note else ""
        lines.insert(end, f"- 状态更新: {status}（{ts}，{actor}）{suffix}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log_activity(root, f"{actor}: {mid} → {status}{suffix}")
        print(f"{mid} → {status}")
        return
    raise SystemExit(f"未找到消息 {mid}（检查 ID 是否拼写正确）")


def main() -> None:
    ap = argparse.ArgumentParser(description="ZCode ↔ Codex 文件协作工具")
    ap.add_argument("--root", default=str(default_root()), help="项目根目录（默认脚本上级的上级）")
    ap.add_argument("--to", choices=["codex", "zcode"], help="追加消息时：写给谁")
    ap.add_argument("--kind", default="task", choices=["task", "done", "review", "bug", "block", "handoff", "decision"])
    ap.add_argument("--title", help="一句话标题")
    ap.add_argument("--body", default="", help="正文（背景/要求/约束）")
    ap.add_argument("--accept", default="", help="验收标准（命令/产物/指标）")
    ap.add_argument("--body-file", default=None, help="从文件读取正文（长消息推荐）")
    ap.add_argument("--accept-file", default=None, help="从文件读取验收标准")
    ap.add_argument("--related", default="", help="关联文件/提交/任务ID")
    ap.add_argument("--owner", default=None, help="发起人（默认按 --to 推断）")
    ap.add_argument("--mark", metavar="MSG-ID", help="更新某条消息的状态")
    ap.add_argument("--status", default="DONE", choices=["OPEN", "ACK", "DONE", "BLOCKED"])
    ap.add_argument("--note", default=None, help="状态附注")
    args = ap.parse_args()

    root = Path(args.root)
    if not (root / "docs" / "collab").exists():
        raise SystemExit(f"{root}/docs/collab 不存在，请确认 --root")

    if args.mark:
        mark(root, args.mark, args.status, args.note)
    else:
        if not args.to or not args.title:
            raise SystemExit("追加消息需要 --to 与 --title")
        body = Path(args.body_file).read_text(encoding="utf-8").strip() if args.body_file else args.body
        accept = Path(args.accept_file).read_text(encoding="utf-8").strip() if args.accept_file else args.accept
        append_msg(root, args.to, args.kind, args.title, body, accept, args.related, args.owner)


if __name__ == "__main__":
    main()
