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
from app.services.context_compaction import prefix_prompt_with_context_summary
from app.services.conversation_context import build_history_for
from app.services.dispatch_prompts import build_agent_hub_tool_guidance
from app.services.dispatch_run_evidence import clear_run_tool_evidence, get_run_tool_evidence
from app.services.event_bus import event_bus
from app.services.pending_bash_commands import pending_bash_commands
from app.services.pending_questions import pending_questions
from app.services.pending_writes import pending_writes
from app.services.project_artifact import maybe_create_project_artifact
from app.services.task_result_report import (
    REPORT_TASK_RESULT_TOOL_NAME,
    is_task_result_report_tool_name,
    read_task_result_report_from_tool_result,
)
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
    override_prompt: str | None = None,
    override_system_prompt: str | None = None,
    override_tool_names: list[str] | None = None,
    require_task_report: bool = False,
) -> str:
    """起一个 run 并**立刻返回 runId**（不等待结束，结果全靠 SSE 推）。

    parent_signal 用于级联中止：父 run 一 abort，子 run 立刻跟着 abort
    （监听器在起跑前挂上；已中止的父 signal 直接把子 run 标成中止态）。

    override_* 供编排使用：子任务/阶段用外部构造的 prompt、system prompt 与工具集；
    require_task_report 时工具集里保证有 report_task_result。
    """
    run_id, _task = _spawn_run(
        conversation_id=conversation_id,
        agent_id=agent_id,
        trigger_message_id=trigger_message_id,
        parent_run_id=parent_run_id,
        parent_signal=parent_signal,
        override_prompt=override_prompt,
        override_system_prompt=override_system_prompt,
        override_tool_names=override_tool_names,
        require_task_report=require_task_report,
    )
    return run_id


def start_run_joined(
    *,
    conversation_id: str,
    agent_id: str,
    trigger_message_id: str,
    parent_run_id: str | None = None,
    parent_signal: AbortSignal | None = None,
    override_prompt: str | None = None,
    override_system_prompt: str | None = None,
    override_tool_names: list[str] | None = None,
    require_task_report: bool = False,
) -> tuple[str, asyncio.Task[Any]]:
    """同 start_run，但把内部 task 一并返回 —— 调度器要 await 子 run 的执行结果。"""
    return _spawn_run(
        conversation_id=conversation_id,
        agent_id=agent_id,
        trigger_message_id=trigger_message_id,
        parent_run_id=parent_run_id,
        parent_signal=parent_signal,
        override_prompt=override_prompt,
        override_system_prompt=override_system_prompt,
        override_tool_names=override_tool_names,
        require_task_report=require_task_report,
    )


def _spawn_run(
    *,
    conversation_id: str,
    agent_id: str,
    trigger_message_id: str,
    parent_run_id: str | None = None,
    parent_signal: AbortSignal | None = None,
    override_prompt: str | None = None,
    override_system_prompt: str | None = None,
    override_tool_names: list[str] | None = None,
    require_task_report: bool = False,
) -> tuple[str, asyncio.Task[Any]]:
    run_id = new_run_id()
    signal = AbortSignal()
    if parent_signal is not None:
        if parent_signal.aborted:
            signal.abort()
        else:
            parent_signal.add_listener(signal.abort)

    active_runs[run_id] = signal

    task = asyncio.create_task(
        _execute_run(
            run_id=run_id,
            signal=signal,
            conversation_id=conversation_id,
            agent_id=agent_id,
            trigger_message_id=trigger_message_id,
            parent_run_id=parent_run_id,
            override_prompt=override_prompt,
            override_system_prompt=override_system_prompt,
            override_tool_names=override_tool_names,
            require_task_report=require_task_report,
        )
    )

    def _cleanup(finished: asyncio.Task) -> None:
        active_runs.pop(run_id, None)
        if parent_signal is not None:
            parent_signal.remove_listener(signal.abort)
        if not finished.cancelled() and finished.exception() is not None:
            logger.error("run %s crashed: %r", run_id, finished.exception())

    task.add_done_callback(_cleanup)
    return run_id, task


def abort_run(run_id: str) -> bool:
    """找到就 abort 并返回 True；找不到（已结束 / 不是本进程起的）返回 False → 路由给 404。"""
    signal = active_runs.get(run_id)
    if signal is None:
        return False
    signal.abort()
    return True


# ─── run 主流程 ─────────────────────────────────────────────


def _run_result(
    run_id: str,
    status: str,
    *,
    error: str | None = None,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """一次 run 的语义结果（子 run 的 await 值；普通 run 由调用方忽略）。"""
    execution = execution or {}
    return {
        "run_id": run_id,
        "status": status,
        "error": error,
        "artifact_ids": list(execution.get("artifact_ids") or []),
        "output_message_ids": list(execution.get("output_message_ids") or []),
        "output_artifacts": dict(execution.get("output_artifacts") or {}),
        "task_report": execution.get("task_report"),
    }


async def _execute_run(
    *,
    run_id: str,
    signal: AbortSignal,
    conversation_id: str,
    agent_id: str,
    trigger_message_id: str,
    parent_run_id: str | None,
    override_prompt: str | None = None,
    override_system_prompt: str | None = None,
    override_tool_names: list[str] | None = None,
    require_task_report: bool = False,
) -> dict[str, Any]:
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
                return _run_result(run_id, "failed", error=f"Agent not found: {agent_id}")

            workspace = await session.scalar(
                select(Workspace).where(Workspace.conversation_id == conversation_id)
            )
            if workspace is None:
                error = f"Workspace not found for conversation: {conversation_id}"
                await _finalize(
                    session,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    status="failed",
                    error=error,
                    output_message_ids=[],
                )
                return _run_result(run_id, "failed", error=error)

            trigger = await session.scalar(
                select(Message).where(
                    Message.id == trigger_message_id,
                    Message.conversation_id == conversation_id,
                )
            )
            if trigger is None:
                error = f"Trigger message not found: {trigger_message_id}"
                await _finalize(
                    session,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    status="failed",
                    error=error,
                    output_message_ids=[],
                )
                return _run_result(run_id, "failed", error=error)

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

            conv = await session.scalar(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            prompt = override_prompt or _extract_text_from_parts(trigger.parts)
            attachments = (
                None
                if override_prompt
                else await _collect_attachments(session, trigger, workspace.root_path)
            )

            if agent.is_orchestrator:
                from app.services.orchestrator_runner import execute_orchestrator_run

                execution = await execute_orchestrator_run(
                    session,
                    run_id=run_id,
                    signal=signal,
                    agent=agent,
                    conv=conv,
                    workspace=workspace,
                    trigger=trigger,
                    user_prompt=prompt,
                    attachments=attachments,
                )
            else:
                execution = await _execute_simple_run(
                    session,
                    run_id=run_id,
                    signal=signal,
                    agent=agent,
                    conv=conv,
                    workspace=workspace,
                    trigger=trigger,
                    prompt=prompt,
                    attachments=attachments,
                    parent_run_id=parent_run_id,
                    override_system_prompt=override_system_prompt,
                    override_tool_names=override_tool_names,
                    require_task_report=require_task_report,
                    skip_history=override_prompt is not None,
                )

            status = "aborted" if signal.aborted else "complete"
            await _finalize(
                session,
                run_id=run_id,
                conversation_id=conversation_id,
                agent_id=agent_id,
                status=status,
                error=None,
                output_message_ids=execution["output_message_ids"],
            )
            return _run_result(run_id, status, execution=execution)

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
                return _run_result(run_id, "aborted")
            await _finalize(
                session,
                run_id=run_id,
                conversation_id=conversation_id,
                agent_id=agent_id,
                status="failed",
                error=message,
                output_message_ids=[],
            )
            return _run_result(run_id, "failed", error=message)


async def _execute_simple_run(
    session: AsyncSession,
    *,
    run_id: str,
    signal: AbortSignal,
    agent: Agent,
    conv: Conversation | None,
    workspace: Workspace,
    trigger: Message,
    prompt: str,
    attachments: list[dict[str, Any]] | None,
    parent_run_id: str | None,
    override_system_prompt: str | None,
    override_tool_names: list[str] | None,
    require_task_report: bool,
    skip_history: bool,
) -> dict[str, Any]:
    """普通 Agent：消费 adapter 事件流；顶层 run 收尾时按写入证据生成 project 产物。"""
    tool_names = list(override_tool_names or agent.tool_names or [])
    if require_task_report and REPORT_TASK_RESULT_TOOL_NAME not in tool_names:
        tool_names.append(REPORT_TASK_RESULT_TOOL_NAME)

    # SDK 类 adapter 不消费 history 数组，上下文摘要改以 prompt 前缀注入
    # （custom adapter 走 build_history_for 的「摘要 + 摘要之后的消息」）。
    # override prompt（编排隔离上下文）不加前缀；失败退化到无前缀，不让 run 崩。
    if agent.adapter_name in ("claude-code", "codex") and not skip_history:
        try:
            prompt = await prefix_prompt_with_context_summary(
                session, trigger.conversation_id, prompt
            )
        except Exception:  # noqa: BLE001 - 摘要前缀是增强，不是依赖
            logger.exception(
                "prefix_prompt_with_context_summary failed; continuing without summary"
            )

    adapter_input = await build_adapter_input(
        session,
        agent=agent,
        conv=conv,
        workspace=workspace,
        run_id=run_id,
        prompt=prompt,
        tool_names=tool_names,
        system_prompt_override=override_system_prompt,
        attachments=attachments,
        conversation_id=trigger.conversation_id,
        exclude_message_id=trigger.id,
        include_history=not skip_history,
    )
    adapter = get_adapter(agent.adapter_name)
    execution = await consume_stream(session, adapter, adapter_input, signal, run_id)

    if parent_run_id is not None:
        # 子 run 的 project 产物由调度器在任务收尾时统一生成
        return execution

    try:
        evidence = get_run_tool_evidence(run_id)
        project_artifact_id = await maybe_create_project_artifact(
            evidence_file_writes=evidence.fileWrites,
            conversation_id=trigger.conversation_id,
            agent_id=agent.id,
        )
        if project_artifact_id:
            execution["artifact_ids"].append(project_artifact_id)
    finally:
        clear_run_tool_evidence(run_id)
    return execution


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


async def build_adapter_input(
    session: AsyncSession,
    *,
    agent: Agent,
    conv: Conversation | None,
    workspace: Workspace,
    run_id: str,
    prompt: str,
    tool_names: list[str],
    system_prompt_override: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    conversation_id: str,
    exclude_message_id: str | None,
    include_history: bool,
) -> AdapterInput:
    """拼 AdapterInput：workspace 块 + system prompt + 工具调用规范 + 历史。

    编排的计划/聚合阶段用 system_prompt_override 换掉 persona；子 Agent 用
    include_history=False 跳过群聊历史（隔离上下文已在 prompt 里）。
    """
    # system prompt：workspace 信息块在前（让 LLM 明确知道自己在哪个目录干活）
    effective_cwd = get_effective_cwd(workspace)
    base_system_prompt = (
        system_prompt_override if system_prompt_override is not None else agent.system_prompt
    )
    system_prompt = _build_workspace_context_block(workspace, effective_cwd) + "\n\n" + base_system_prompt

    # 按实际工具集补一段调用规范（计划阶段 / 本地项目模式 / 各工具的用法约束）
    tool_guidance = build_agent_hub_tool_guidance(agent.adapter_name, tool_names, workspace.mode)
    if tool_guidance:
        system_prompt += "\n\n" + tool_guidance

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
    if agent.adapter_name == "custom" and include_history:
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
                conversation_id,
                exclude_message_id=exclude_message_id,
                token_budget=history_budget,
            )
        except Exception:  # noqa: BLE001 - 历史是增强，不是依赖
            logger.exception("build_history_for failed; continuing without history")

    return AdapterInput(
        agentId=agent.id,
        conversationId=conversation_id,
        runId=run_id,
        prompt=prompt,
        workspacePath=effective_cwd,
        systemPrompt=system_prompt,
        apiKey=api_key,
        apiBaseUrl=agent.api_base_url,
        modelId=agent.model_id,
        toolNames=list(tool_names),
        attachments=attachments,
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


def read_artifact_handoff_result(result: Any) -> dict[str, str] | None:
    """工具结果里的 artifact 交接声明：`{artifactId, outputKey}`（outputKey 去空白后非空）。"""
    if not isinstance(result, dict):
        return None
    artifact_id = result.get("artifactId")
    output_key = result.get("outputKey")
    if not isinstance(artifact_id, str) or not isinstance(output_key, str):
        return None
    if not output_key.strip():
        return None
    return {"artifactId": artifact_id, "outputKey": output_key}


async def consume_stream(
    session: AsyncSession,
    adapter: AgentPlatformAdapter,
    adapter_input: AdapterInput,
    signal: AbortSignal,
    run_id: str,
    on_tool_call: Any = None,
) -> dict[str, Any]:
    """消费 adapter 事件：**每条先落库、再广播**。

    落库可能派生新事件（artifact.create → 注入 artifact_ref part），派生事件同样按
    「先落库、再广播」的顺序补发。额外收集执行结果：产物 id 列表、按 outputKey
    归位的产物映射、report_task_result 上报的任务报告。

    on_tool_call(event) 可返回 `{"stop": True, "result": ..., "isError": ...}` 提前
    结束本轮（计划阶段拿到 plan_tasks 就收工）：合成 tool.result / message.end
    同样先落库再广播，然后中断消费。
    """
    parts_buffer: dict[str, list[Any]] = {}
    artifact_ids: list[str] = []
    output_message_ids: list[str] = []
    output_artifacts: dict[str, str] = {}
    output_key_by_artifact_id: dict[str, str] = {}
    tool_name_by_call_id: dict[str, str] = {}
    task_report: Any = None
    current_message_id: str | None = None

    async for event in adapter.stream(adapter_input, signal):
        if isinstance(event, MessageStartEvent):
            current_message_id = event.messageId
        if isinstance(event, ToolCallEvent):
            tool_name_by_call_id[event.callId] = event.toolName

        derived = await _persist_event(
            session,
            event,
            parts_buffer=parts_buffer,
            output_message_ids=output_message_ids,
            artifact_ids=artifact_ids,
            run_id=run_id,
            agent_id=adapter_input.agentId,
            current_message_id=current_message_id,
        )
        event_bus.publish(event)
        for extra in derived:
            event_bus.publish(extra)

        if isinstance(event, ArtifactCreateEvent):
            output_key = output_key_by_artifact_id.get(event.artifact.id)
            if output_key:
                output_artifacts[output_key] = event.artifact.id

        if isinstance(event, MessageEndEvent):
            current_message_id = None

        if isinstance(event, ToolResultEvent):
            tool_name = tool_name_by_call_id.get(event.callId)
            if tool_name and not event.isError and is_task_result_report_tool_name(tool_name):
                report = read_task_result_report_from_tool_result(event.result)
                if report:
                    task_report = report
            handoff = read_artifact_handoff_result(event.result)
            if handoff:
                output_key_by_artifact_id[handoff["artifactId"]] = handoff["outputKey"]

        if isinstance(event, ToolCallEvent) and on_tool_call is not None:
            control = on_tool_call(event)
            if isinstance(control, dict) and control.get("stop"):
                if "result" in control:
                    result_event = ToolResultEvent(
                        conversationId=event.conversationId,
                        timestamp=now_ms(),
                        messageId=event.messageId,
                        callId=event.callId,
                        result=control["result"],
                        isError=bool(control.get("isError", False)),
                    )
                    await _persist_event(
                        session,
                        result_event,
                        parts_buffer=parts_buffer,
                        output_message_ids=output_message_ids,
                        artifact_ids=artifact_ids,
                        run_id=run_id,
                        agent_id=adapter_input.agentId,
                        current_message_id=current_message_id,
                    )
                    event_bus.publish(result_event)

                end_event = MessageEndEvent(
                    conversationId=event.conversationId,
                    timestamp=now_ms(),
                    messageId=event.messageId,
                )
                await _persist_event(
                    session,
                    end_event,
                    parts_buffer=parts_buffer,
                    output_message_ids=output_message_ids,
                    artifact_ids=artifact_ids,
                    run_id=run_id,
                    agent_id=adapter_input.agentId,
                    current_message_id=current_message_id,
                )
                event_bus.publish(end_event)
                current_message_id = None
                break

    return {
        "artifact_ids": artifact_ids,
        "output_message_ids": output_message_ids,
        "output_artifacts": output_artifacts,
        "task_report": task_report,
        "current_message_id": current_message_id,
    }


async def _persist_event(
    session: AsyncSession,
    event: StreamEvent,
    *,
    parts_buffer: dict[str, list[Any]],
    output_message_ids: list[str],
    artifact_ids: list[str],
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
        artifact_ids.append(event.artifact.id)
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
