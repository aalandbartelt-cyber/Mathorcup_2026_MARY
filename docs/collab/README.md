# ZCode ↔ Codex 文件协作协议

> 目的：ZCode（Windows 侧，辅助）与 Codex（WSL 侧，核心开发）不直接对话，全部通过仓库里的 Markdown 文件交接工作。
> 所有文件在同一个 git 仓库内，天然可审计、可回溯。

## 0.0 角色

| 角色 | 位置 | 职责 | 边界 |
|---|---|---|---|
| Codex | WSL | 实现、训练、推理、评估、论文 | 不替 dsh 做建模主张 |
| ZCode | Windows | 环境、数据、文档、复核 | **不扮演 dsh** |
| dsh（DeepSeek Harness） | 独立 AI | 建模推理、选型、结果解读、评审 | 结论以 decision 消息进入，经 Codex 落地 |

## 0. 给 Codex 的启动指令（用户复制粘贴即可）

```
请阅读 docs/collab/README.md 与 docs/collab/board.md，
然后检查 docs/collab/to_codex.md 中状态为 OPEN 的消息。
按协议处理：在你自己的发件箱 docs/collab/to_zcode.md 追加回复，
并把原消息的状态行改为 ACK / DONE / BLOCKED（只改状态行，不改正文）。
```

## 0.1 使用频率原则（2026-10-21 起）

本通道**赛前不用于日常往返**，只在三种情况使用：
1. 任务交接（一次把背景、产物路径、验收写全）；
2. 阻塞或需要对方决策；
3. 赛后复盘归档。
日常进度只更新 `board.md`，不发消息；能一句话在对话里说清的，不写文件。

## 1. 文件职责（单一写入者原则）

| 文件 | 谁写 | 谁读 | 说明 |
|---|---|---|---|
| `board.md` | 双方（各自维护区） | 双方 | 任务板：当前任务、状态、阻塞、活动日志 |
| `to_codex.md` | **只有 ZCode** | Codex | ZCode 发给 Codex 的消息；Codex 只改“状态”行 |
| `to_zcode.md` | **只有 Codex** | ZCode | Codex 发给 ZCode 的消息；ZCode 只改“状态”行 |
| `decisions.md` | 双方追加 | 双方 | 已定事项（编号 D-xxx），只追加不删除 |
| `handoffs/` | 交接方 | 接手方 | 大任务交接文档（可选，按任务建 `<MSG-ID>.md`） |

**铁律**：不删别人的消息、不改别人的正文；只追加、只改状态行。

## 2. 消息格式（模板）

```markdown
## MSG-YYYYMMDD-HHMM-xx | 类型 | 状态
- 发起: ZCode | Codex
- 接收: Codex | ZCode
- 时间: YYYY-MM-DD HH:MM
- 关联: 文件/提交/任务ID（可多个）

**标题**: 一句话说明要做什么

**内容**:
背景、要做什么、约束、相关文件路径。

**验收**:
可检查的完成标准（命令、产物路径、指标）。

**回复**: （由接收方填写，指向 to_xxx.md 中的回复消息 ID）
```

- 类型：`task` 任务 / `done` 完成回报 / `block` 阻塞上报 / `review` 审查 / `bug` 缺陷 / `handoff` 交接 / `decision` 决策征询
- 状态机：`OPEN → ACK → DONE`；异常 `BLOCKED`（写原因与选项）；暂不执行 `PAUSED`（条件满足后再恢复）

## 3. 开工 / 收工流程

### Codex（WSL 侧，核心开发）
1. `git pull`（先同步）
2. 读 `board.md`，确认当前任务与 owner
3. 读 `to_codex.md`，找 `OPEN` 消息 → 状态改 `ACK`
4. 干活；完成后在 `to_zcode.md` 追加回复（类型 `done`），原消息改 `DONE`
5. 更新 `board.md` 自己维护区的任务状态与活动日志
6. `git add docs/collab && git commit -m "collab(MSG-ID): ..." && git push`

### ZCode（Windows 侧，辅助）
1. `git pull`
2. 读 `to_zcode.md` 的 `OPEN` 消息 → `ACK`
3. 处理（环境、数据、复核、文档）→ 在 `to_codex.md` 回复 → 原消息 `DONE`
4. 维护 `board.md` 的 ZCode 区与活动日志
5. 同样提交推送

## 4. 状态与阻塞规则

- 收到消息 **15 分钟内** 至少把状态改成 `ACK`（表示“看到了，在排队”）。
- 预计超过 **2 小时** 完不成，改 `BLOCKED`，写清：卡在哪、已尝试什么、需要对方做什么、可选方案。
- 一条消息只对应一件事；多件事拆多条消息。
- 同一文件不要两边同时大改；需要改同一文件时，先发 `task` 消息约定“谁先改、改哪段”。

## 5. Git 约定

- 提交信息：`collab(MSG-ID): 一句话`，例如 `collab(MSG-20261021-1030-a1): 完成 577 张推理`
- 每次处理完消息立即提交推送，避免积压。
- 若产生合并冲突：**保留双方消息，重新编号**，在 `board.md` 活动日志记一行；绝不通过删除对方内容解决冲突。

## 6. 目录

```
docs/collab/
├── README.md          # 本协议
├── board.md           # 任务板（双方各自维护区 + 活动日志）
├── to_codex.md        # ZCode → Codex（ZCode 写，Codex 改状态）
├── to_zcode.md        # Codex → ZCode（Codex 写，ZCode 改状态）
├── decisions.md       # 已定事项 D-001…
└── handoffs/          # 大任务交接文档（可选）
```

## 7. 快捷脚本

```bash
# 追加一条消息（自动生成 ID/时间/模板）
python scripts/collab_msg.py --to codex --kind task --title "跑通练习2推理" --body "..." --accept "输出 outputs/ct_pred.csv"

# 改状态（接收方执行）
python scripts/collab_msg.py --mark MSG-20261021-1030-a1 --status DONE --note "产物在 outputs/ct_pred.csv"
```
