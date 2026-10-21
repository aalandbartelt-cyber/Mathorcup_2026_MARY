# 协作任务板

> 更新时间：2026-10-21 10:30（ZCode）
> 规则：双方只维护自己的区块；活动日志只追加。协议见 `docs/collab/README.md`。

## 当前阶段

- 目标：2026 年 MathorCup 大数据竞赛（预计 **2026-10-23 18:00** 开赛）
- 状态：环境/文档/模板/校验脚本已就绪；等待 Codex 回报 WSL 侧环境与可用时间

## 角色

| 角色 | 位置 | 职责 |
|---|---|---|
| Codex | WSL（VS Code Remote-WSL 扩展） | 核心开发：代码、实验、调试、论文 LaTeX |
| ZCode | Windows 宿主 | 辅助：环境运维、数据整理、文档、复核、替补 |

## ZCode 维护区

| 任务ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| Z-001 | 搭建协作协议与消息模板 | DONE | commit e30f2f3 之后新增 collab/ |
| Z-002 | 比赛执行清单/结果模板/校验脚本 | DONE | docs/比赛执行清单.md、scripts/check_result.py |
| Z-003 | 等待 Codex 首次回报 | OPEN | 见 to_codex.md MSG-20261021-1030-a1 |
| Z-004 | 按 Codex 需求补环境/数据 | PENDING | 收到 BLOCKED 或 task 后启动 |

## Codex 维护区

| 任务ID | 任务 | 状态 | 备注 |
|---|---|---|---|
| C-001 | 阅读协议并 ACK MSG-001 | OPEN | 启动指令见 README §0 |
| C-002 | 登记 WSL 环境状态 | OPEN | conda/GPU/ultralytics/可用时段 |
| C-003 | （可选）练习2 的 577 张推理跑通 | PENDING | 由 Codex 决定优先级 |

## 阻塞

| 消息ID | 阻塞点 | 需要谁 | 状态 |
|---|---|---|---|
| — | — | — | — |

## 活动日志

- [2026-10-21 10:30] ZCode: 建立 docs/collab 协议、任务板与首条消息 MSG-001
