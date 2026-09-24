"""会话与消息的领域逻辑：会话增删改查 / 置顶 / 归档 / 重命名、消息落库、清空历史、发消息并起 run。

LLM 适配器不在这一层 —— 这里只负责决定「谁该回复」并把活交给 AgentRunner。
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (
    Agent,
    AgentRun,
    Artifact,
    Attachment,
    ContextSummary,
    Conversation,
    Message,
    Workspace,
)
from app.errors import ConflictError, NotFoundError, ServiceError
from app.schemas.entities import MessageOut
from app.schemas.events import (
    MessageAddedEvent,
    MessageRecord,
    MessageRemovedEvent,
    MessageUsageEvent,
)
from app.security.workspace_utils import is_path_safe
from app.services import deploy_command_service
from app.services.agent_runner import abort_run, start_run
from app.services.event_bus import event_bus
from app.services.pending_dispatch_plans import pending_dispatch_plans
from app.utils.ids import new_conversation_id, new_message_id, new_workspace_id
from app.utils.time import now_ms

_IS_WINDOWS = os.name == "nt"

# pin 上限（与 web/src/shared/constants.ts 的 PIN_LIMIT_PER_CONVERSATION 一致）
PIN_LIMIT_PER_CONVERSATION = 5


def _attach_workspace_meta(conv: Conversation, ws: Workspace | None) -> Conversation:
    """把 workspace 的 mode / boundPath 挂到会话行上，供 ConversationWithMetaOut 读取。

    前端多处要显示「本地工作目录」标识，listConversations 一次 JOIN 出来避免 lazy fetch。
    """
    conv.workspace_mode = ws.mode if ws else "sandbox"
    conv.workspace_bound_path = ws.bound_path if ws else None
    return conv


async def _workspace_for(session: AsyncSession, conversation_id: str) -> Workspace | None:
    return await session.scalar(
        select(Workspace).where(Workspace.conversation_id == conversation_id)
    )


async def _rm_dir_with_retry(target: str) -> None:
    """Windows 上 EBUSY/EPERM/ENOTEMPTY 走指数退避（100/300/900ms）。"""
    retryable = {"EBUSY", "EPERM", "ENOTEMPTY"}
    for attempt in range(1, 4):
        try:
            shutil.rmtree(target, ignore_errors=True)
            return
        except OSError as err:
            errno_name = getattr(err, "errno", None)
            name = os.strerror(errno_name) if errno_name else ""
            if name not in retryable or attempt == 3:
                raise
            await asyncio.sleep(0.1 * (3 ** (attempt - 1)))


# ─── 创建会话 ────────────────────────────────────────────
def _default_title_for(names: list[str]) -> str:
    if len(names) == 1:
        return f"与 {names[0]} 的对话"
    return " / ".join(names)


async def create_conversation(session: AsyncSession, args: dict[str, Any]) -> Conversation:
    agent_ids: list[str] = args["agent_ids"]
    mode: str = args["mode"]

    if len(agent_ids) == 0:
        raise ServiceError("At least one agent is required")
    if mode == "single" and len(agent_ids) != 1:
        raise ServiceError("Single conversation requires exactly one agent")
    if mode == "group" and len(agent_ids) < 2:
        raise ServiceError("Group conversation requires at least two agents")

    found = list(await session.scalars(select(Agent).where(Agent.id.in_(agent_ids))))
    found_ids = {a.id for a in found}
    if len(found) != len(agent_ids):
        missing = [i for i in agent_ids if i not in found_ids]
        raise ServiceError(f"Agents not found: {', '.join(missing)}")

    # 标题按请求顺序取名，保证确定性（直接按 DB 查询结果取名时，组内顺序并不确定）
    name_by_id = {a.id: a.name for a in found}
    names = [name_by_id[i] for i in agent_ids]

    workspace_mode = "sandbox"
    resolved_bound_path: str | None = None
    raw_bound = (args.get("bound_path") or "").strip()
    if raw_bound:
        if _IS_WINDOWS and not (len(raw_bound) > 2 and raw_bound[1] == ":" and raw_bound[2] in "\\/"):
            if not raw_bound.startswith("\\\\"):
                raise ServiceError(
                    f"boundPath must start with a drive letter (e.g. D:\\projects\\foo) "
                    f"on Windows: {raw_bound}"
                )
        # realpath 会解析符号链接，拿到真实路径。
        # expanduser 是 Python 侧多加的一点方便（前端目录选择器只会给绝对路径，
        # 所以实际请求不受影响；~ 展开只会让本该 400 的输入变成合法路径）。
        candidate = os.path.realpath(os.path.expanduser(raw_bound))
        if not os.path.isabs(candidate):
            raise ServiceError("boundPath must be absolute")
        if not os.path.exists(candidate):
            raise ServiceError(f"Path does not exist: {candidate}")
        if not os.path.isdir(candidate):
            raise ServiceError(f"Not a directory: {candidate}")
        if not os.access(candidate, os.R_OK | os.W_OK):
            raise ServiceError(f"Not readable/writable: {candidate}")
        if not is_path_safe(candidate):
            raise ServiceError(f"Path is not allowed (system / sensitive directory): {candidate}")
        workspace_mode = "local"
        resolved_bound_path = candidate

    settings = get_settings()
    now = now_ms()
    conversation_id = new_conversation_id()
    workspace_id = new_workspace_id()
    root_path = str(settings.workspaces_root / conversation_id)

    # 内部 sandbox 目录无论 mode 都要建，用于 attachments 等内部文件
    Path(root_path).mkdir(parents=True, exist_ok=True)

    title = args.get("title") or _default_title_for(names)

    conversation = Conversation(
        id=conversation_id,
        title=title,
        mode=mode,
        agent_ids=agent_ids,
        pinned_message_ids=[],
        bookmarked_message_ids=[],
        archived=False,
        pinned_at=None,
        fs_write_approval_mode="review",
        created_at=now,
        updated_at=now,
    )
    workspace = Workspace(
        id=workspace_id,
        conversation_id=conversation_id,
        root_path=root_path,
        mode=workspace_mode,
        bound_path=resolved_bound_path,
        created_at=now,
    )
    session.add_all([conversation, workspace])
    await session.commit()

    return _attach_workspace_meta(conversation, workspace)


# ─── 列出会话 ────────────────────────────────────────────
async def list_conversations(session: AsyncSession) -> list[Conversation]:
    # pinnedAt desc nulls last + updatedAt desc：置顶在前，相互按 pinnedAt 倒序；未置顶按活跃时间
    conversations = list(
        await session.scalars(
            select(Conversation).order_by(
                Conversation.pinned_at.desc(), Conversation.updated_at.desc()
            )
        )
    )
    if not conversations:
        return []

    workspaces = list(
        await session.scalars(
            select(Workspace).where(
                Workspace.conversation_id.in_([c.id for c in conversations])
            )
        )
    )
    ws_by_conv = {w.conversation_id: w for w in workspaces}

    return [_attach_workspace_meta(c, ws_by_conv.get(c.id)) for c in conversations]


# ─── 置顶 / 归档 ─────────────────────────────────────────
async def toggle_pin_conversation(session: AsyncSession, conversation_id: str) -> Conversation:
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if conv is None:
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    conv.pinned_at = None if conv.pinned_at else now_ms()
    await session.commit()
    return _attach_workspace_meta(conv, await _workspace_for(session, conversation_id))


async def toggle_archive_conversation(session: AsyncSession, conversation_id: str) -> Conversation:
    """归档是会话级元操作，不更新 updatedAt（不应顶到列表前），与 togglePin 一致。"""
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if conv is None:
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    conv.archived = not conv.archived
    await session.commit()
    return _attach_workspace_meta(conv, await _workspace_for(session, conversation_id))


# ─── 重命名 / 审批模式 ───────────────────────────────────
async def rename_conversation(
    session: AsyncSession, conversation_id: str, title: str
) -> Conversation:
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if conv is None:
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    trimmed = title.strip()
    if not trimmed:
        raise ServiceError("Title cannot be empty")
    if len(trimmed) > 100:
        raise ServiceError("Title too long (max 100)")

    conv.title = trimmed
    conv.updated_at = now_ms()
    await session.commit()
    return _attach_workspace_meta(conv, await _workspace_for(session, conversation_id))


async def set_conversation_approval_mode(
    session: AsyncSession, conversation_id: str, mode: str
) -> Conversation:
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if conv is None:
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    conv.fs_write_approval_mode = mode
    conv.updated_at = now_ms()
    await session.commit()
    return _attach_workspace_meta(conv, await _workspace_for(session, conversation_id))


# ─── 添加 Agent 到现有会话 ──────────────────────────────
async def add_agents_to_conversation(
    session: AsyncSession, conversation_id: str, agent_ids: list[str]
) -> Conversation:
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if conv is None:
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    found = list(await session.scalars(select(Agent).where(Agent.id.in_(agent_ids))))
    found_ids = {a.id for a in found}
    if len(found) != len(agent_ids):
        missing = [i for i in agent_ids if i not in found_ids]
        raise ServiceError(f"Agents not found: {', '.join(missing)}")

    merged = list(dict.fromkeys([*conv.agent_ids, *agent_ids]))
    conv.agent_ids = merged
    conv.mode = "group" if len(merged) >= 2 else "single"
    conv.updated_at = now_ms()
    await session.commit()
    return _attach_workspace_meta(conv, await _workspace_for(session, conversation_id))


# ─── 列出消息 ────────────────────────────────────────────
async def list_messages(session: AsyncSession, conversation_id: str) -> list[Message]:
    return list(
        await session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
    )


# ─── 删除会话 ────────────────────────────────────────────
async def delete_conversation(session: AsyncSession, conversation_id: str) -> None:
    workspace = await _workspace_for(session, conversation_id)

    # 删 DB：依赖 ON DELETE CASCADE 级联清 messages / artifacts / workspaces / attachments / runs
    result = await session.execute(
        sa_delete(Conversation).where(Conversation.id == conversation_id)
    )
    await session.commit()

    if result.rowcount == 0:
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    if workspace:
        try:
            await _rm_dir_with_retry(workspace.root_path)
        except OSError as err:
            print(f"[deleteConversation] failed to remove workspace dir {workspace.root_path}: {err}")


# ─── 清空会话历史 ────────────────────────────────────────
async def clear_conversation_history(session: AsyncSession, conversation_id: str) -> dict[str, Any]:
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if conv is None:
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    active_runs = await session.scalar(
        select(func.count())
        .select_from(AgentRun)
        .where(
            AgentRun.conversation_id == conversation_id,
            AgentRun.status.in_(["queued", "running"]),
        )
    )
    if active_runs:
        raise ConflictError("Cannot clear conversation history while agent runs are active")

    deleted_message_count = await session.scalar(
        select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)
    )
    deleted_run_count = await session.scalar(
        select(func.count()).select_from(AgentRun).where(AgentRun.conversation_id == conversation_id)
    )
    deleted_summary_count = await session.scalar(
        select(func.count())
        .select_from(ContextSummary)
        .where(ContextSummary.conversation_id == conversation_id)
    )

    now = now_ms()
    await session.execute(
        sa_delete(ContextSummary).where(ContextSummary.conversation_id == conversation_id)
    )
    await session.execute(sa_delete(Message).where(Message.conversation_id == conversation_id))
    await session.execute(sa_delete(AgentRun).where(AgentRun.conversation_id == conversation_id))
    await session.execute(
        sa_update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(pinned_message_ids=[], bookmarked_message_ids=[], updated_at=now)
    )
    await session.commit()

    await session.refresh(conv)
    return {
        "conversation": _attach_workspace_meta(
            conv, await _workspace_for(session, conversation_id)
        ),
        "deletedMessageCount": deleted_message_count or 0,
        "deletedRunCount": deleted_run_count or 0,
        "deletedSummaryCount": deleted_summary_count or 0,
    }


# ─── 发消息 ──────────────────────────────────────────────
async def send_message(session: AsyncSession, args: dict[str, Any]) -> dict[str, Any]:
    conversation_id: str = args["conversation_id"]
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if conv is None:
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    now = now_ms()
    message_id = new_message_id()

    parts: list[dict[str, Any]] = []
    content: str = args.get("content") or ""
    if content.strip():
        parts.append({"type": "text", "content": content})

    attachment_ids = args.get("attachment_ids") or []
    if attachment_ids:
        rows = list(
            await session.scalars(select(Attachment).where(Attachment.id.in_(attachment_ids)))
        )
        for row in rows:
            if row.conversation_id != conversation_id:
                continue  # 防越权引用其他会话的附件
            parts.append(
                {
                    "type": "image_attachment" if row.kind == "image" else "file_attachment",
                    "attachmentId": row.id,
                    "fileName": row.file_name,
                    "size": row.size,
                    "mimeType": row.mime_type,
                }
            )

    mentioned_agent_ids = args.get("mentioned_agent_ids") or []
    parent_message_id = args.get("parent_message_id")

    message = Message(
        id=message_id,
        conversation_id=conversation_id,
        role="user",
        agent_id=None,
        parts=parts,
        status="complete",
        parent_message_id=parent_message_id,
        mentioned_agent_ids=mentioned_agent_ids,
        run_id=None,
        usage=None,
        created_at=now,
    )
    session.add(message)
    conv.updated_at = now
    await session.commit()

    # 先广播用户消息（其它已连接客户端立即插入），**再**起 run —— 顺序不能反
    event_bus.publish(
        MessageAddedEvent(
            conversationId=conversation_id,
            timestamp=now,
            message=_message_record(message),
        )
    )

    # 「部署」指令：只认纯文本单 part、无 @、无附件、无 parent 的消息，避免误判正常提问
    deploy_intent = (
        deploy_command_service.parse_deploy_command(content)
        if len(parts) == 1
        and not parent_message_id
        and not mentioned_agent_ids
        and not attachment_ids
        else None
    )
    if deploy_intent is not None:
        deploy = await deploy_command_service.handle_deploy_command(
            session, conversation_id, deploy_intent.get("artifact_id"), after_created_at=now
        )
        return {
            "messageId": message_id,
            "runIds": [],
            "messages": [deploy["message"]],
            "deploy": deploy,
        }

    agents_in_conv = await _agents_in_conversation(session, list(conv.agent_ids))
    responder_ids = decide_responders(conv, list(mentioned_agent_ids), agents_in_conv)

    # 起 run 但**不等待**：202 立刻返回 runIds，结果全靠 SSE 推
    run_ids = [
        start_run(
            conversation_id=conversation_id,
            agent_id=agent_id,
            trigger_message_id=message_id,
        )
        for agent_id in responder_ids
    ]

    return {"messageId": message_id, "runIds": run_ids}


async def revise_dispatch_plan(
    session: AsyncSession, conversation_id: str, plan_id: str, feedback: str
) -> dict[str, Any]:
    """对话式修改待审计划：反馈作为一条 user 消息落库并广播（进对话、跨端可见），
    再交回正在等待的编排 run 重排。不触发新 run（现有 run 会自己续上并发出新计划）。"""
    pending = pending_dispatch_plans.get(plan_id)
    if pending is None or pending.conversationId != conversation_id:
        return {"ok": False, "error": "Pending dispatch plan not found"}

    message_id = new_message_id()
    now = now_ms()
    parts: list[dict[str, Any]] = [{"type": "text", "content": feedback}]

    message = Message(
        id=message_id,
        conversation_id=conversation_id,
        role="user",
        agent_id=None,
        parts=parts,
        status="complete",
        parent_message_id=None,
        mentioned_agent_ids=[],
        run_id=None,
        usage=None,
        created_at=now,
    )
    session.add(message)
    await session.execute(
        sa_update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(updated_at=now)
    )
    await session.commit()

    event_bus.publish(
        MessageAddedEvent(
            conversationId=conversation_id,
            timestamp=now,
            message=_message_record(message),
        )
    )

    ok = pending_dispatch_plans.revise(plan_id, feedback)
    return (
        {"ok": True}
        if ok
        else {"ok": False, "error": "Failed to revise pending dispatch plan"}
    )


def _message_record(row: Message) -> MessageRecord:
    """ORM 行 → 事件里的 MessageRecord（camelCase 字段，由 snake_case 列映射而来）。"""
    return MessageRecord(
        id=row.id,
        conversationId=row.conversation_id,
        role=row.role,  # type: ignore[arg-type]
        agentId=row.agent_id,
        parts=row.parts or [],
        status=row.status,  # type: ignore[arg-type]
        parentMessageId=row.parent_message_id,
        mentionedAgentIds=list(row.mentioned_agent_ids or []),
        runId=row.run_id,
        usage=MessageUsageEvent(**row.usage) if row.usage else None,
        createdAt=row.created_at,
    )


async def _agents_in_conversation(session: AsyncSession, agent_ids: list[str]) -> list[Agent]:
    """按会话里的 agent 顺序返回（直接一次 IN 查询的话，顺序由 SQLite 决定，不保证）。

    刻意按 `conv.agent_ids` 排序：顺序会影响「群聊无 @ 时选哪个 orchestrator」，
    按会话顺序是确定性的，跨次重启结果一致。
    """
    if not agent_ids:
        return []
    rows = list(await session.scalars(select(Agent).where(Agent.id.in_(agent_ids))))
    by_id = {row.id: row for row in rows}
    return [by_id[i] for i in agent_ids if i in by_id]


def decide_responders(conv: Conversation, mentions: list[str], agents_in_conv: list[Agent]) -> list[str]:
    """哪些 agent 该被这条消息触发。

    - 单聊：会话里的**全部** agent（不过滤 @）
    - 群聊带 @：按 @ 顺序，且必须在该会话里（不在的静默丢弃；可能得到空列表 → 不起 run）
    - 群聊不带 @：只找 **一个** orchestrator；没有 orchestrator → 不起 run
    """
    if conv.mode == "single":
        return list(conv.agent_ids)
    if mentions:
        return [m for m in mentions if m in conv.agent_ids]
    orchestrator = next((a for a in agents_in_conv if a.is_orchestrator), None)
    return [orchestrator.id] if orchestrator is not None else []


# ─── 书签 / Pin 消息 ─────────────────────────────────────
async def _conversation_or_404(session: AsyncSession, conversation_id: str) -> Conversation:
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if conv is None:
        raise NotFoundError(f"Conversation not found: {conversation_id}")
    return conv


async def _message_in_conversation(
    session: AsyncSession, conversation_id: str, message_id: str
) -> Message:
    row = await session.scalar(
        select(Message).where(
            Message.id == message_id, Message.conversation_id == conversation_id
        )
    )
    if row is None:
        raise NotFoundError(f"Message not found in conversation: {message_id}")
    return row


def _toggle_in_list(current: list[str], message_id: str) -> tuple[list[str], bool]:
    """返回 (新列表, 本次是「加入」还是「移除」)。"""
    if message_id in current:
        return [i for i in current if i != message_id], False
    return [*current, message_id], True


async def toggle_bookmarked_message(
    session: AsyncSession, conversation_id: str, message_id: str
) -> dict[str, Any]:
    """UI 书签：只用于导航定位/高亮，不进 LLM 上下文（pin 才进）。无条数上限。"""
    conv = await _conversation_or_404(session, conversation_id)
    await _message_in_conversation(session, conversation_id, message_id)

    next_ids, bookmarked = _toggle_in_list(list(conv.bookmarked_message_ids or []), message_id)
    conv.bookmarked_message_ids = next_ids
    conv.updated_at = now_ms()
    await session.commit()
    return {"bookmarkedMessageIds": next_ids, "bookmarked": bookmarked}


async def toggle_pinned_message(
    session: AsyncSession, conversation_id: str, message_id: str
) -> dict[str, Any]:
    """Pin = 注入 LLM 长期上下文（agent 每次拼上下文都带上）。

    与书签的差异：有 PIN_LIMIT_PER_CONVERSATION 上限；不更新 updatedAt
    （pin 不算「会话活跃」，不应把会话顶到侧栏最前）。
    """
    conv = await _conversation_or_404(session, conversation_id)
    await _message_in_conversation(session, conversation_id, message_id)

    current = list(conv.pinned_message_ids or [])
    if message_id not in current and len(current) >= PIN_LIMIT_PER_CONVERSATION:
        raise ServiceError("PIN_LIMIT_EXCEEDED")
    next_ids, pinned = _toggle_in_list(current, message_id)

    conv.pinned_message_ids = next_ids
    await session.commit()
    return {"pinnedMessageIds": next_ids, "pinned": pinned}


# ─── 撤回 / 重新生成 / 编辑重发 ───────────────────────────
def _artifact_ids_in(messages: list[Message]) -> list[str]:
    """从 parts 里收集 artifact_ref 指向的产物 id（撤回时要级联删掉）。"""
    ids: list[str] = []
    for message in messages:
        for part in message.parts or []:
            if part.get("type") == "artifact_ref" and part.get("artifactId"):
                ids.append(part["artifactId"])
    return sorted(set(ids))


async def _abort_runs_since(
    session: AsyncSession, conversation_id: str, since_ms: int, *, inclusive: bool
) -> None:
    """中止时间窗内还在跑的 run，并留 500ms 让它们的 finalize 落库。

    这 500ms 是有意义的：abort 之后 runner 还会补 `[已中止]` 文本 part / 死消息，
    不等就会漏删，客户端就会看到一个「撤回后残留的半截回复」。
    """
    comparison = AgentRun.started_at >= since_ms if inclusive else AgentRun.started_at > since_ms
    runs = list(
        await session.scalars(
            select(AgentRun).where(
                AgentRun.conversation_id == conversation_id,
                comparison,
                AgentRun.status == "running",
            )
        )
    )
    if not runs:
        return
    for row in runs:
        abort_run(row.id)
    await asyncio.sleep(0.5)


async def _delete_message_window(
    session: AsyncSession, conversation_id: str, since_ms: int, *, inclusive: bool
) -> list[Message]:
    """删除时间窗内的消息 + 它们的产物 + run，返回被删掉的消息行（含 parts）。"""
    message_window = Message.created_at >= since_ms if inclusive else Message.created_at > since_ms
    messages = list(
        await session.scalars(
            select(Message).where(Message.conversation_id == conversation_id, message_window)
        )
    )
    run_window = (
        AgentRun.started_at >= since_ms if inclusive else AgentRun.started_at > since_ms
    )
    runs = list(
        await session.scalars(
            select(AgentRun).where(AgentRun.conversation_id == conversation_id, run_window)
        )
    )
    artifact_ids = _artifact_ids_in(messages)

    if messages:
        await session.execute(sa_delete(Message).where(Message.id.in_([m.id for m in messages])))
    if artifact_ids:
        await session.execute(sa_delete(Artifact).where(Artifact.id.in_(artifact_ids)))
    if runs:
        await session.execute(sa_delete(AgentRun).where(AgentRun.id.in_([r.id for r in runs])))
    await session.commit()

    # 先落库再广播：其它已连接客户端据此实时移除被撤回的消息与产物
    event_bus.publish(
        MessageRemovedEvent(
            conversationId=conversation_id,
            timestamp=now_ms(),
            messageIds=[m.id for m in messages],
            artifactIds=artifact_ids,
        )
    )
    return messages


async def withdraw_latest_user_message(
    session: AsyncSession, conversation_id: str, message_id: str
) -> dict[str, Any]:
    """撤回最后一条 user 消息：连同它之后的所有消息 / 产物 / run 一起删掉。

    时间窗用 `>=`（含触发消息本身）；只允许撤回**最后一条** user 消息，
    避免「撤一条老的、留下上下文空洞」。
    """
    message = await session.scalar(
        select(Message).where(
            Message.id == message_id, Message.conversation_id == conversation_id
        )
    )
    if message is None:
        raise NotFoundError(f"Message not found: {message_id}")
    if message.role != "user":
        raise ServiceError("Only user messages can be withdrawn")

    latest_user = await session.scalar(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.role == "user")
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    if latest_user is None or latest_user.id != message_id:
        raise ServiceError("Only the latest user message can be withdrawn")

    await _abort_runs_since(session, conversation_id, message.created_at, inclusive=True)
    deleted = await _delete_message_window(
        session, conversation_id, message.created_at, inclusive=True
    )
    return {
        "deletedMessageIds": [m.id for m in deleted],
        "deletedArtifactIds": _artifact_ids_in(deleted),
    }


async def regenerate_latest_response(session: AsyncSession, conversation_id: str) -> dict[str, Any]:
    """重新生成最后一次回复：保留最后一条 user 消息，删它之后的回复并重新起 run。"""
    conv = await _conversation_or_404(session, conversation_id)

    latest_user = await session.scalar(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.role == "user")
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    if latest_user is None:
        raise ServiceError("No user message to regenerate from")

    # 时间窗用 `>`（保留触发消息本身）
    await _abort_runs_since(session, conversation_id, latest_user.created_at, inclusive=False)
    deleted = await _delete_message_window(
        session, conversation_id, latest_user.created_at, inclusive=False
    )

    agents_in_conv = await _agents_in_conversation(session, list(conv.agent_ids))
    responders = decide_responders(
        conv, list(latest_user.mentioned_agent_ids or []), agents_in_conv
    )
    run_ids = [
        start_run(
            conversation_id=conversation_id,
            agent_id=agent_id,
            trigger_message_id=latest_user.id,
        )
        for agent_id in responders
    ]
    return {
        "deletedMessageIds": [m.id for m in deleted],
        "deletedArtifactIds": _artifact_ids_in(deleted),
        "triggerMessageId": latest_user.id,
        "runIds": run_ids,
    }


async def edit_and_resend_latest_user_message(
    session: AsyncSession, conversation_id: str, message_id: str, new_content: str
) -> dict[str, Any]:
    """编辑最后一条 user 消息：撤回原消息，再用新内容重发（保留原 @、parent、附件）。"""
    trimmed = new_content.strip()
    if not trimmed:
        raise ServiceError("Content cannot be empty")

    original = await session.scalar(
        select(Message).where(
            Message.id == message_id, Message.conversation_id == conversation_id
        )
    )
    if original is None:
        raise NotFoundError(f"Message not found: {message_id}")
    if original.role != "user":
        raise ServiceError("Only user messages can be edited")

    attachment_ids = [
        part["attachmentId"]
        for part in original.parts or []
        if part.get("type") in ("image_attachment", "file_attachment") and part.get("attachmentId")
    ]

    withdrawn = await withdraw_latest_user_message(session, conversation_id, message_id)

    sent = await send_message(
        session,
        {
            "conversation_id": conversation_id,
            "content": trimmed,
            "mentioned_agent_ids": list(original.mentioned_agent_ids or []),
            "parent_message_id": original.parent_message_id,
            "attachment_ids": attachment_ids or None,
        },
    )

    new_row = await session.scalar(select(Message).where(Message.id == sent["messageId"]))
    if new_row is None:
        raise ServiceError("New message disappeared after insert")

    return {
        **withdrawn,
        "newMessage": MessageOut.model_validate(new_row).model_dump(by_alias=True),
        "runIds": sent["runIds"],
    }
