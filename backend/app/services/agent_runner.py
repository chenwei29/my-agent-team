"""一条 run 的生命周期（P2 最小版）：起 run、驱动 adapter、逐事件落库、广播 StreamEvent、支持中止。

顺序固定（别调整）：

    insertRun(status='running') → publish(run.start)
      → consume(adapter.stream)：每条事件 **先落库、再 publish**
      → finalize：run 行落终态 → 该 run 下仍 streaming 的 message 落终态
        → （failed/aborted 时）补未完成 tool_result + 推 `[已中止]`/`[失败]` 文本 part
        → 更新 conversation.updatedAt → publish(run.end)

必须记住的几条（踩过就明白为什么）：
- **先写库再推事件**：客户端收到事件后可能立刻重新拉取，库里必须已经是终态。
- **每个 run 用独立 session**：SQLAlchemy async session 不能跨 task 共享。
- **run 路径不开事务**：每条命令各自提交，中间状态对其它请求可见，SSE 契约依赖这一点。
- **run.end.error 在成功与中止时都不带**：不带就是 JSON 里直接没这个键，而非 null。
- **message.end 之后 run 仍可能被 abort**：mock adapter 中止时照样吐 `message.end`，
  所以会看到「message=complete + run=aborted + 追加 [已中止] part」这种组合，属正常行为。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.custom import CustomAgentAdapter
from app.adapters.mock import MockAdapter
from app.adapters.types import AdapterInput, AgentPlatformAdapter
from app.db.models import Agent, AgentRun, Attachment, Conversation, Message, Workspace
from app.db.session import SessionLocal
from app.errors import ServiceError
from app.schemas.events import (
    ArtifactCreateEvent,
    DeployStatusEvent,
    MessageEndEvent,
    MessageStartEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunEndEvent,
    RunStartEvent,
    RunUsageEventWrapper,
    MessageUsageEventWrapper,
    StreamEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.security.workspace_utils import get_effective_cwd
from app.services import settings_service
from app.services.conversation_context import build_history_for
from app.services.event_bus import event_bus
from app.services.pending_bash_commands import pending_bash_commands
from app.services.pending_questions import pending_questions
from app.services.pending_writes import pending_writes
from app.utils.abort import AbortSignal
from app.utils.ids import new_run_id
from app.utils.model_registry import estimate_tokens, get_model_limits
from app.utils.time import now_ms

logger = logging.getLogger(__name__)

# 模块级 run 注册表：只放「还在跑的」run，abort 只查这里，不查 DB
active_runs: dict[str, AbortSignal] = {}

_MOCK_ADAPTER = MockAdapter()
_CUSTOM_ADAPTER = CustomAgentAdapter()


def get_adapter(adapter_name: str) -> AgentPlatformAdapter:
    if adapter_name == "mock":
        return _MOCK_ADAPTER
    if adapter_name == "custom":
        return _CUSTOM_ADAPTER
    # claude-code / codex 是后续阶段的活。这里明确报错（run 落 failed +
    # 前端看到失败提示），不要退化成 mock 假装成功。
    raise ServiceError(f"Adapter not implemented yet: {adapter_name} (planned for a later phase)")


# ─── 启动 / 中止 ────────────────────────────────────────────


def start_run(
    *,
    conversation_id: str,
    agent_id: str,
    trigger_message_id: str,
    parent_run_id: str | None = None,
    parent_signal: AbortSignal | None = None,
) -> str:
    """起一个 run 并**立刻返回 runId**（不等待结束，结果全靠 SSE 推）。"""
    run_id = new_run_id()
    signal = AbortSignal()
    if parent_signal is not None and parent_signal.aborted:
        signal.abort()

    active_runs[run_id] = signal

    task = asyncio.create_task(
        _execute_run(
            run_id=run_id,
            signal=signal,
            conversation_id=conversation_id,
            agent_id=agent_id,
            trigger_message_id=trigger_message_id,
            parent_run_id=parent_run_id,
        )
    )

    def _cleanup(finished: asyncio.Task) -> None:
        active_runs.pop(run_id, None)
        if not finished.cancelled() and finished.exception() is not None:
            logger.error("run %s crashed: %r", run_id, finished.exception())

    task.add_done_callback(_cleanup)
    return run_id


def abort_run(run_id: str) -> bool:
    """找到就 abort 并返回 True；找不到（已结束 / 不是本进程起的）返回 False → 路由给 404。"""
    signal = active_runs.get(run_id)
    if signal is None:
        return False
    signal.abort()
    return True


# ─── run 主流程 ─────────────────────────────────────────────


async def _execute_run(
    *,
    run_id: str,
    signal: AbortSignal,
    conversation_id: str,
    agent_id: str,
    trigger_message_id: str,
    parent_run_id: str | None,
) -> None:
    async with SessionLocal() as session:
        try:
            agent = await session.scalar(select(Agent).where(Agent.id == agent_id))
            if agent is None:
                # 预检失败发生在 insertRun 之前 —— 会推 run.end，但库里根本没有 run 行
                await _finalize(
                    session,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    status="failed",
                    error=f"Agent not found: {agent_id}",
                    output_message_ids=[],
                )
                return

            workspace = await session.scalar(
                select(Workspace).where(Workspace.conversation_id == conversation_id)
            )
            if workspace is None:
                await _finalize(
                    session,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    status="failed",
                    error=f"Workspace not found for conversation: {conversation_id}",
                    output_message_ids=[],
                )
                return

            trigger = await session.scalar(
                select(Message).where(
                    Message.id == trigger_message_id,
                    Message.conversation_id == conversation_id,
                )
            )
            if trigger is None:
                await _finalize(
                    session,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    status="failed",
                    error=f"Trigger message not found: {trigger_message_id}",
                    output_message_ids=[],
                )
                return

            # 1) run 行先落库
            session.add(
                AgentRun(
                    id=run_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    trigger_message_id=trigger_message_id,
                    status="running",
                    error=None,
                    parent_run_id=parent_run_id,
                    usage=None,
                    started_at=now_ms(),
                    finished_at=None,
                )
            )
            await session.commit()

            # 2) 再推 run.start（timestamp 与 started_at 是两次独立的 now 取值）
            event_bus.publish(
                RunStartEvent(
                    conversationId=conversation_id,
                    timestamp=now_ms(),
                    runId=run_id,
                    agentId=agent_id,
                    triggerMessageId=trigger_message_id,
                    parentRunId=parent_run_id,
                )
            )

            adapter = get_adapter(agent.adapter_name)
            conv = await session.scalar(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            adapter_input = await _build_adapter_input(
                session,
                agent=agent,
                conv=conv,
                workspace=workspace,
                trigger=trigger,
                run_id=run_id,
            )

            result = await _consume_stream(session, adapter, adapter_input, signal, run_id)

            if signal.aborted:
                await _finalize(
                    session,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    status="aborted",
                    error=None,
                    output_message_ids=result["output_message_ids"],
                )
            else:
                await _finalize(
                    session,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    status="complete",
                    error=None,
                    output_message_ids=result["output_message_ids"],
                )

        except Exception as err:  # noqa: BLE001 - run 内部任何异常都转成 failed/aborted
            logger.exception("run %s failed", run_id)
            await session.rollback()
            message = str(err)
            if signal.aborted:
                await _finalize(
                    session,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    status="aborted",
                    error=None,
                    output_message_ids=[],
                )
            else:
                await _finalize(
                    session,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    status="failed",
                    error=message,
                    output_message_ids=[],
                )


# ─── Adapter 输入构造 ─────────────────────────────────────


# 群聊里别 agent 的发言在历史中以 `[名字] ` 前缀的 user 消息出现，
# 这段说明让当前 agent 正确解读前缀语义，不把别人的话当成自己的输出。
_GROUP_CHAT_SYSTEM_NOTE = "\n".join(
    [
        "## 群聊上下文",
        "当前会话是多 Agent 群聊。历史里其他成员（含 Orchestrator）的发言，会以 `[成员名] ` 前缀的 user 消息出现。",
        "- 带 `[名字]` 前缀的 user 消息是别的成员说的，不是你自己的输出，也不是用户的直接指令——按需参考即可。",
        "- 不带前缀的 user 消息才是用户本人发给群里的话。",
        "- 历史里的产物只折叠成 `[产物: 标题 (id=...)]` 占位；需要完整内容时用 read_artifact 按 id 获取，不要凭占位臆测。",
    ]
)


async def _build_adapter_input(
    session: AsyncSession,
    *,
    agent: Agent,
    conv: Conversation | None,
    workspace: Workspace,
    trigger: Message,
    run_id: str,
) -> AdapterInput:
    prompt = _extract_text_from_parts(trigger.parts)

    # system prompt：workspace 信息块在前（让 LLM 明确知道自己在哪个目录干活）
    effective_cwd = get_effective_cwd(workspace)
    system_prompt = _build_workspace_context_block(workspace, effective_cwd) + "\n\n" + agent.system_prompt

    # Key 三层解析：agents.api_key > app_settings > 环境变量。
    # 只在 per-agent 字段为空时才注入全局配置，避免覆盖用户的精细配置。
    # openai-compatible 没有「全局」key 可回退：key 与 endpoint 成对，必须 per-agent 填。
    api_key = agent.api_key
    if not api_key and agent.model_provider and agent.model_provider != "openai-compatible":
        api_key = await settings_service.get_effective_api_key(
            session, _settings_provider_key(agent.model_provider)
        )

    # 跨 run 历史注入：只有 custom adapter 消费（ClaudeCode / Codex 走
    # SDK session 续接；mock 忽略）。失败退化到「无历史」，不让 run 崩。
    history: list[dict[str, Any]] = []
    if agent.adapter_name == "custom":
        if conv is not None and len(conv.agent_ids or []) > 1:
            system_prompt += "\n\n" + _GROUP_CHAT_SYSTEM_NOTE
        try:
            limits = get_model_limits(agent.model_provider, agent.model_id)
            prompt_estimate = (
                estimate_tokens(system_prompt) + estimate_tokens(prompt) + 512  # 安全余量
            )
            history_budget = max(0, limits.context_window - limits.output_reserve - prompt_estimate)
            history = await build_history_for(
                session,
                agent.id,
                trigger.conversation_id,
                exclude_message_id=trigger.id,
                token_budget=history_budget,
            )
        except Exception:  # noqa: BLE001 - 历史是增强，不是依赖
            logger.exception("build_history_for failed; continuing without history")

    return AdapterInput(
        agentId=agent.id,
        conversationId=trigger.conversation_id,
        runId=run_id,
        prompt=prompt,
        workspacePath=effective_cwd,
        systemPrompt=system_prompt,
        apiKey=api_key,
        apiBaseUrl=agent.api_base_url,
        modelId=agent.model_id,
        toolNames=list(agent.tool_names or []),
        attachments=await _collect_attachments(session, trigger, workspace.root_path),
        history=history or None,
        customConfig=(
            {
                "modelProvider": agent.model_provider,
                "supportsVision": agent.supports_vision,
            }
            if agent.adapter_name == "custom" and agent.model_provider and agent.model_id
            else None
        ),
    )


def _settings_provider_key(model_provider: str) -> str:
    """model_provider 枚举 → app_settings 里的 key 名（仅 volcano-ark 不同名）。"""
    return "ark" if model_provider == "volcano-ark" else model_provider


def _build_workspace_context_block(workspace: Workspace, cwd: str) -> str:
    """给 LLM 注入「我在哪个目录工作」的 XML 块。

    解决 LLM 看到工具描述里的 "inside the workspace" 时误以为是隔离沙箱、
    声称「无法访问本地文件」的问题（即使 workspace 实际绑定了真实项目）。
    """
    if workspace.mode == "local":
        note = (
            "This directory is the user's REAL local project on their machine. "
            "Files inside it are their actual code. When you use fs_list / fs_read / "
            "fs_write / bash, you are reading and modifying real files — be careful. "
            "You CAN access these files directly via the workspace tools; do not tell "
            "the user you cannot access local files."
        )
        mode = "local"
    else:
        note = (
            "This is an isolated sandbox directory (under .agenthub-data/). "
            "It is NOT the user's real codebase. Files you write here are only "
            "visible inside this conversation."
        )
        mode = "sandbox"
    return "\n".join(
        [
            "<workspace_info>",
            f"  <cwd>{cwd}</cwd>",
            f"  <mode>{mode}</mode>",
            f"  <note>{note}</note>",
            "</workspace_info>",
        ]
    )


async def _collect_attachments(
    session: AsyncSession, trigger: Message, workspace_root: str
) -> list[dict[str, Any]] | None:
    """触发消息里的附件 → adapter 附件列表（绝对路径 + mime），查不到的跳过。"""
    from pathlib import Path

    attachment_ids = [
        p.get("attachmentId")
        for p in trigger.parts or []
        if isinstance(p, dict) and p.get("type") in ("image_attachment", "file_attachment")
    ]
    if not attachment_ids:
        return None

    rows = list(await session.scalars(select(Attachment).where(Attachment.id.in_(attachment_ids))))
    return [
        {
            "id": row.id,
            "fileName": row.file_name,
            "mimeType": row.mime_type,
            "kind": row.kind,
            "absPath": str(Path(workspace_root) / row.file_path),
        }
        for row in rows
    ]


async def _consume_stream(
    session: AsyncSession,
    adapter: AgentPlatformAdapter,
    adapter_input: AdapterInput,
    signal: AbortSignal,
    run_id: str,
) -> dict[str, Any]:
    """消费 adapter 事件：**每条先落库、再广播**。

    落库可能派生新事件（artifact.create → 注入 artifact_ref part），派生事件同样按
    「先落库、再广播」的顺序补发。
    """
    parts_buffer: dict[str, list[Any]] = {}
    output_message_ids: list[str] = []
    current_message_id: str | None = None

    async for event in adapter.stream(adapter_input, signal):
        if isinstance(event, MessageStartEvent):
            current_message_id = event.messageId
        derived = await _persist_event(
            session,
            event,
            parts_buffer=parts_buffer,
            output_message_ids=output_message_ids,
            run_id=run_id,
            agent_id=adapter_input.agentId,
            current_message_id=current_message_id,
        )
        event_bus.publish(event)
        for extra in derived:
            event_bus.publish(extra)

    return {"output_message_ids": output_message_ids, "current_message_id": current_message_id}


async def _persist_event(
    session: AsyncSession,
    event: StreamEvent,
    *,
    parts_buffer: dict[str, list[Any]],
    output_message_ids: list[str],
    run_id: str,
    agent_id: str,
    current_message_id: str | None = None,
) -> list[StreamEvent]:
    """落库一条事件；返回需要补发的派生事件（通常是 part.start）。"""
    derived: list[StreamEvent] = []
    if isinstance(event, RunUsageEventWrapper):
        await session.execute(
            sa_update(AgentRun)
            .where(AgentRun.id == event.runId)
            .values(usage=event.usage.model_dump())
        )
        await session.commit()

    elif isinstance(event, MessageUsageEventWrapper):
        await session.execute(
            sa_update(Message)
            .where(Message.id == event.messageId)
            .values(usage=event.usage.model_dump())
        )
        await session.commit()

    elif isinstance(event, MessageStartEvent):
        parts_buffer[event.messageId] = []
        output_message_ids.append(event.messageId)
        session.add(
            Message(
                id=event.messageId,
                conversation_id=event.conversationId,
                role="agent",
                agent_id=agent_id,
                parts=[],
                status="streaming",
                parent_message_id=None,
                mentioned_agent_ids=[],
                run_id=run_id,
                usage=None,
                created_at=event.timestamp,
            )
        )
        await session.commit()

    elif isinstance(event, PartStartEvent):
        parts = parts_buffer.setdefault(event.messageId, [])
        while len(parts) <= event.partIndex:  # 稀疏下标：中间留空洞，序列化成 JSON 就是 null
            parts.append(None)
        parts[event.partIndex] = event.part.model_dump()
        await _write_parts(session, event.messageId, parts)

    elif isinstance(event, PartDeltaEvent):
        parts = parts_buffer.get(event.messageId)
        if parts is not None and event.partIndex < len(parts):
            part = parts[event.partIndex]
            if isinstance(part, dict) and _delta_matches_part(event.delta.type, part.get("type")):
                part["content"] = part.get("content", "") + event.delta.text
                await _write_parts(session, event.messageId, parts)

    elif isinstance(event, ToolCallEvent):
        parts = parts_buffer.setdefault(event.messageId, [])
        parts.append(
            {
                "type": "tool_use",
                "callId": event.callId,
                "toolName": event.toolName,
                "args": event.args,
            }
        )
        await _write_parts(session, event.messageId, parts)

    elif isinstance(event, ToolResultEvent):
        parts = parts_buffer.setdefault(event.messageId, [])
        parts.append(
            {
                "type": "tool_result",
                "callId": event.callId,
                "result": event.result,
                "isError": event.isError,
            }
        )
        await _write_parts(session, event.messageId, parts)

    elif isinstance(event, ArtifactCreateEvent):
        # 工具产出的产物除了 tool_result，还要在消息里挂一个 artifact_ref part，
        # 否则聊天流里只有一行工具日志、看不到产物卡片
        if current_message_id is not None:
            extra = await _append_part(
                session,
                parts_buffer,
                current_message_id,
                {"type": "artifact_ref", "artifactId": event.artifact.id},
                event,
            )
            if extra is not None:
                derived.append(extra)

    elif isinstance(event, DeployStatusEvent):
        extra = await _append_part(
            session,
            parts_buffer,
            event.messageId,
            {"type": "deploy_status", "deployment": event.deployment},
            event,
        )
        if extra is not None:
            derived.append(extra)

    elif isinstance(event, MessageEndEvent):
        await session.execute(
            sa_update(Message).where(Message.id == event.messageId).values(status="complete")
        )
        await session.commit()
        parts_buffer.pop(event.messageId, None)

    # 其它事件（part.end / message.added / artifact.update / dispatch.*）在落库侧是 no-op
    return derived


async def _append_part(
    session: AsyncSession,
    parts_buffer: dict[str, list[Any]],
    message_id: str,
    part: dict[str, Any],
    source: StreamEvent,
) -> PartStartEvent | None:
    """往消息末尾追加一个 part 并落库；消息已收尾（不在 buffer 里）时什么都不做。"""
    parts = parts_buffer.get(message_id)
    if parts is None:
        return None

    part_index = len(parts)
    parts.append(part)
    await _write_parts(session, message_id, parts)
    return PartStartEvent(
        conversationId=source.conversationId,
        timestamp=now_ms(),
        messageId=message_id,
        partIndex=part_index,
        part=part,
    )


def _delta_matches_part(delta_type: str, part_type: Any) -> bool:
    return (delta_type, part_type) in {
        ("text.append", "text"),
        ("thinking.append", "thinking"),
        ("code.append", "code"),
    }


async def _write_parts(session: AsyncSession, message_id: str, parts: list[Any]) -> None:
    """每次 part.start / part.delta / tool.* 都把整个 parts 数组写回（逐 delta 一次全量 UPDATE）。"""
    await session.execute(
        sa_update(Message).where(Message.id == message_id).values(parts=list(parts))
    )
    await session.commit()


# ─── 收尾 ───────────────────────────────────────────────────


async def _finalize(
    session: AsyncSession,
    *,
    run_id: str,
    conversation_id: str,
    agent_id: str,
    status: str,  # 'complete' | 'failed' | 'aborted'
    error: str | None,
    output_message_ids: list[str],
) -> None:
    finished_at = now_ms()

    if status in ("failed", "aborted"):
        # run 终止：清掉它名下所有挂起的审批项（工具侧的 abort listener 负责
        # 在飞的 await，这里兜底 register→attach 之间竞态漏掉的）
        pending_writes.cancel_for_run(run_id)
        pending_questions.cancel_for_run(run_id)
        pending_bash_commands.cancel_for_run(run_id)

    if status in ("failed", "aborted"):
        await _persist_unresolved_tool_failures(
            session,
            run_id=run_id,
            conversation_id=conversation_id,
            status=status,
            error=error,
            timestamp=finished_at,
        )

    await session.execute(
        sa_update(AgentRun)
        .where(AgentRun.id == run_id)
        .values(status=status, finished_at=finished_at, error=error)
    )
    message_status = {"complete": "complete", "aborted": "aborted"}.get(status, "error")
    await session.execute(
        sa_update(Message)
        .where(Message.run_id == run_id, Message.status == "streaming")
        .values(status=message_status)
    )

    if status in ("failed", "aborted"):
        await _emit_error_visualisation(
            session,
            run_id=run_id,
            conversation_id=conversation_id,
            agent_id=agent_id,
            status=status,
            error=error,
            output_message_ids=output_message_ids,
            timestamp=finished_at,
        )

    await session.execute(
        sa_update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(updated_at=finished_at)
    )
    await session.commit()

    event_bus.publish(
        RunEndEvent(
            conversationId=conversation_id,
            timestamp=finished_at,
            runId=run_id,
            status=status,  # type: ignore[arg-type]
            error=error,
        )
    )


def _unresolved_tool_failure_text(status: str, error: str | None) -> str:
    if status == "aborted":
        return "工具调用未完成：本次运行已中止。"
    return f"工具调用未完成：本次运行失败。{error}" if error else "工具调用未完成：本次运行失败。"


async def _persist_unresolved_tool_failures(
    session: AsyncSession,
    *,
    run_id: str,
    conversation_id: str,
    status: str,
    error: str | None,
    timestamp: int,
) -> None:
    """给没有配对 tool_result 的 tool_use 补一条失败 result，避免前端工具卡片永远转圈。"""
    messages = list(await session.scalars(select(Message).where(Message.run_id == run_id)))
    result = _unresolved_tool_failure_text(status, error)

    for message in messages:
        parts = list(message.parts or [])
        completed = {p.get("callId") for p in parts if isinstance(p, dict) and p.get("type") == "tool_result"}
        missing: list[str] = []
        for part in parts:
            if not isinstance(part, dict) or part.get("type") != "tool_use":
                continue
            call_id = part.get("callId")
            if call_id in completed:
                continue
            parts.append(
                {"type": "tool_result", "callId": call_id, "result": result, "isError": True}
            )
            completed.add(call_id)
            missing.append(call_id)

        if not missing:
            continue

        await session.execute(
            sa_update(Message).where(Message.id == message.id).values(parts=parts)
        )
        await session.commit()
        for call_id in missing:
            event_bus.publish(
                ToolResultEvent(
                    conversationId=conversation_id,
                    timestamp=timestamp,
                    messageId=message.id,
                    callId=call_id,
                    result=result,
                    isError=True,
                )
            )


async def _emit_error_visualisation(
    session: AsyncSession,
    *,
    run_id: str,
    conversation_id: str,
    agent_id: str,
    status: str,
    error: str | None,
    output_message_ids: list[str],
    timestamp: int,
) -> None:
    """中止/失败时在聊天里留一条看得见的痕迹：追加到已产出的消息，或新建 msg_err_<runId>。"""
    error_text = "[已中止]" if status == "aborted" else f"[失败] {error or '未知错误'}"

    if output_message_ids:
        last_id = output_message_ids[-1]
        message = await session.scalar(select(Message).where(Message.id == last_id))
        if message is not None:
            parts = list(message.parts or [])
            parts.append({"type": "text", "content": error_text})
            await session.execute(
                sa_update(Message).where(Message.id == last_id).values(parts=parts)
            )
            await session.commit()
            event_bus.publish(
                PartStartEvent(
                    conversationId=conversation_id,
                    timestamp=timestamp,
                    messageId=last_id,
                    partIndex=len(parts) - 1,
                    part={"type": "text", "content": error_text},  # type: ignore[arg-type]
                )
            )
            return

    # 没有可追加的消息：新建一条合成错误消息（id 故意不是 msg_<12 base62>，而是 msg_err_<runId>）
    message_id = f"msg_err_{run_id}"
    session.add(
        Message(
            id=message_id,
            conversation_id=conversation_id,
            role="agent",
            agent_id=agent_id,
            parts=[{"type": "text", "content": error_text}],
            status="error",
            parent_message_id=None,
            mentioned_agent_ids=[],
            run_id=run_id,
            usage=None,
            created_at=timestamp,
        )
    )
    await session.commit()

    event_bus.publish(
        MessageStartEvent(
            conversationId=conversation_id,
            timestamp=timestamp,
            messageId=message_id,
            agentId=agent_id,
            runId=run_id,
        )
    )
    event_bus.publish(
        PartStartEvent(
            conversationId=conversation_id,
            timestamp=timestamp,
            messageId=message_id,
            partIndex=0,
            part={"type": "text", "content": error_text},  # type: ignore[arg-type]
        )
    )
    event_bus.publish(
        MessageEndEvent(conversationId=conversation_id, timestamp=timestamp, messageId=message_id)
    )


def _extract_text_from_parts(parts: list[Any]) -> str:
    """把 text part 拼起来当 prompt（换行连接）。"""
    chunks = [
        p.get("content", "")
        for p in parts or []
        if isinstance(p, dict) and p.get("type") == "text"
    ]
    return "\n".join(chunks)
