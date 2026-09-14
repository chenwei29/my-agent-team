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

from app.adapters.mock import MockAdapter
from app.adapters.types import AdapterInput, AgentPlatformAdapter
from app.db.models import Agent, AgentRun, Conversation, Message, Workspace
from app.db.session import SessionLocal
from app.errors import ServiceError
from app.schemas.events import (
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
from app.services.event_bus import event_bus
from app.utils.abort import AbortSignal
from app.utils.ids import new_run_id
from app.utils.time import now_ms

logger = logging.getLogger(__name__)

# 模块级 run 注册表：只放「还在跑的」run，abort 只查这里，不查 DB
active_runs: dict[str, AbortSignal] = {}

_MOCK_ADAPTER = MockAdapter()


def get_adapter(adapter_name: str) -> AgentPlatformAdapter:
    if adapter_name == "mock":
        return _MOCK_ADAPTER
    # P2 只有 mock：custom / claude-code / codex 分别是 P3 / P8 的活。
    # 这里明确报错（run 落 failed + 前端看到失败提示），不要退化成 mock 假装成功。
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
            adapter_input = AdapterInput(
                agentId=agent_id,
                conversationId=conversation_id,
                runId=run_id,
                prompt=_extract_text_from_parts(trigger.parts),
                workspacePath=workspace.root_path,
                systemPrompt=agent.system_prompt,
                apiKey=agent.api_key,
                apiBaseUrl=agent.api_base_url,
                modelId=agent.model_id,
                toolNames=list(agent.tool_names or []),
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


async def _consume_stream(
    session: AsyncSession,
    adapter: AgentPlatformAdapter,
    adapter_input: AdapterInput,
    signal: AbortSignal,
    run_id: str,
) -> dict[str, Any]:
    """消费 adapter 事件：**每条先落库、再广播**。"""
    parts_buffer: dict[str, list[Any]] = {}
    output_message_ids: list[str] = []
    current_message_id: str | None = None

    async for event in adapter.stream(adapter_input, signal):
        if isinstance(event, MessageStartEvent):
            current_message_id = event.messageId
        await _persist_event(
            session,
            event,
            parts_buffer=parts_buffer,
            output_message_ids=output_message_ids,
            run_id=run_id,
            agent_id=adapter_input.agentId,
        )
        event_bus.publish(event)

    return {"output_message_ids": output_message_ids, "current_message_id": current_message_id}


async def _persist_event(
    session: AsyncSession,
    event: StreamEvent,
    *,
    parts_buffer: dict[str, list[Any]],
    output_message_ids: list[str],
    run_id: str,
    agent_id: str,
) -> None:
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

    elif isinstance(event, MessageEndEvent):
        await session.execute(
            sa_update(Message).where(Message.id == event.messageId).values(status="complete")
        )
        await session.commit()
        parts_buffer.pop(event.messageId, None)

    # 其它事件（part.end / message.added / artifact.* / deploy.* / dispatch.*）在 P2 落库侧是 no-op


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
