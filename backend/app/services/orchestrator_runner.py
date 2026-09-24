"""编排 run 的阶段机：PLAN → REVIEW（待审门禁）→ EXECUTE（DAG 波次）→ 补救轮 → AGGREGATE。

职责边界：
- prompt 拼装在 dispatch_prompts（纯函数），波次调度在 dispatch_scheduler（注入 ChildTaskRuntime），
  本模块负责把阶段串起来：起 plan 流、挂待审计划、批准后喂 DAG、按结果决定补救轮、最后聚合发言；
- 子 Agent 经 AgentChildRuntime 起子 run（override_prompt + require_task_report），
  隔离上下文（依赖闭包产物 / 最近对话 / pin / 摘要）在 build_sub_agent_prompt 里拼好；
- 计划阶段的 plan_tasks 调用是收工信号：consume_stream 的 on_tool_call 拿到它就合成
  tool.result + message.end 并停流，不让模型继续闲聊烧 token。

轮次上限 MAX_DISPATCH_ROUNDS：每轮「plan → 审批 → 执行」；本轮有未完成任务或写冲突时
进补救轮（上一轮结果摘要喂回 plan 阶段）；用户拒绝计划：首轮直接结束，补救轮进聚合。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Agent, Artifact, Conversation, Message, Workspace
from app.db.session import SessionLocal
from app.schemas.dispatch import (
    DispatchPlanItem,
    ReplanConflictView,
    ReplanTaskView,
    RunCommandEvidence,
    RunToolEvidence,
    TaskResultReport,
)
from app.schemas.events import DispatchPlanEvent, ToolCallEvent
from app.security.workspace_utils import assert_path_within_workspace, get_effective_cwd
from app.services.conversation_context import get_latest_context_summary, render_conversation_summary_block
from app.services.dispatch_plan import (
    build_replan_context,
    build_revise_context,
    collect_dependency_closure,
    compile_dispatch_plan,
    extract_plan_tasks_tool_args,
    parse_dispatch_plan_tool_args,
    should_replan,
    validate_dispatch_plan,
)
from app.services.dispatch_prompts import (
    ASK_USER_TOOL_NAME,
    SubAgentPromptData,
    build_aggregate_prompt,
    build_orchestrator_aggregate_prompt,
    build_orchestrator_plan_prompt,
    build_sub_agent_prompt,
)
from app.services.dispatch_run_evidence import get_run_tool_evidence, record_run_command
from app.services.dispatch_scheduler import (
    DagContext,
    DispatchTaskResult,
    PreparedCommand,
    VerificationCommandResult,
    execute_dag,
    run_child_task,
)
from app.services.event_bus import event_bus
from app.services.pending_dispatch_plans import PlanReviewOutcome, pending_dispatch_plans
from app.services.project_artifact import maybe_create_project_artifact
from app.utils.abort import AbortSignal
from app.utils.time import now_ms

from app.services.agent_runner import (
    build_adapter_input,
    consume_stream,
    get_adapter,
    start_run_joined,
)

# 补救轮上限（含首轮）；子任务重试上限在 dispatch_scheduler.MAX_CHILD_TASK_ATTEMPTS
MAX_DISPATCH_ROUNDS = 4
# 计划阶段允许的工具（只读侦察 + 提问 + 拆解），其余工具不给
ORCHESTRATOR_PLAN_ALLOWED_TOOLS = {
    "plan_tasks",
    ASK_USER_TOOL_NAME,
    "fs_list",
    "fs_read",
    "read_artifact",
    "read_attachment",
}
# 子 Agent 上下文里「最近对话」与「已有产物」的条数上限
SUB_AGENT_CONTEXT_RECENT_LIMIT = 5


def _ensure_includes(names: list[str], name: str) -> list[str]:
    return names if name in names else [*names, name]


def compile_and_validate_dispatch_plan(
    raw_plan: list[DispatchPlanItem],
    available_agents: list[Any],
    orchestrator_agent_id: str,
    resolved_external_tasks: list[DispatchPlanItem] | None = None,
) -> list[DispatchPlanItem]:
    """编译（inputs → 依赖）+ 语义校验；坏计划抛 ValueError，文案给用户看。"""
    plan, _inferred = compile_dispatch_plan(raw_plan)
    validate_dispatch_plan(plan, available_agents, orchestrator_agent_id, resolved_external_tasks)
    return plan


def to_replan_views(
    plan_items_by_id: dict[str, DispatchPlanItem],
    merged_results: dict[str, DispatchTaskResult],
) -> list[ReplanTaskView]:
    """merged_results + 计划 → 重规划视图（should_replan / build_replan_context 的输入）。"""
    views: list[ReplanTaskView] = []
    for task_id, item in plan_items_by_id.items():
        result = merged_results.get(task_id)
        status = result.status if result is not None and result.status else "skipped"
        views.append(
            ReplanTaskView(
                taskId=task_id,
                agentId=item.agentId,
                status=status,
                error=result.error if result is not None else None,
            )
        )
    return views


def to_replan_conflicts(conflicts: list[Any]) -> list[ReplanConflictView]:
    return [
        ReplanConflictView(path=conflict.path, taskIds=[w.taskId for w in conflict.contributors])
        for conflict in conflicts
    ]


# ─── 子任务执行的生产依赖面 ─────────────────────────────────


class AgentChildRuntime:
    """ChildTaskRuntime 的生产实现：起子 run、拼隔离上下文、补跑命令、生成 project 产物。"""

    def __init__(self, conversation_id: str, workspace: Workspace) -> None:
        self.conversation_id = conversation_id
        self.workspace = workspace

    async def build_sub_agent_prompt(
        self,
        task: DispatchPlanItem,
        resolved_inputs: list[Any],
        upstream: dict[str, DispatchTaskResult],
        plan: list[DispatchPlanItem],
    ) -> str:
        # 传递依赖闭包的产物都算上游上下文（审查任务要看得到 PRD / 设计稿，不只直接上游）
        upstream_artifact_ids: set[str] = set()
        for dep_id in collect_dependency_closure(plan, task.id):
            dep_result = upstream.get(dep_id)
            if dep_result is not None:
                upstream_artifact_ids.update(dep_result.artifact_ids)

        async with SessionLocal() as session:
            upstream_artifacts: list[Artifact] = []
            if upstream_artifact_ids:
                upstream_artifacts = list(
                    await session.scalars(
                        select(Artifact).where(Artifact.id.in_(upstream_artifact_ids))
                    )
                )

            existing_all = list(
                await session.scalars(
                    select(Artifact)
                    .where(Artifact.conversation_id == self.conversation_id)
                    .order_by(Artifact.created_at.desc())
                )
            )
            existing = [
                artifact
                for artifact in existing_all
                if artifact.id not in upstream_artifact_ids
            ][:SUB_AGENT_CONTEXT_RECENT_LIMIT]

            latest_summary = await get_latest_context_summary(session, self.conversation_id)
            conditions = [
                Message.conversation_id == self.conversation_id,
                Message.status == "complete",
            ]
            if latest_summary is not None:
                # 摘要覆盖过的部分不再进最近对话（摘要块在 prompt 里已给）
                conditions.append(
                    Message.created_at > (latest_summary.covered_until_created_at or 0)
                )
            recent_rows = list(
                await session.scalars(
                    select(Message)
                    .where(*conditions)
                    .order_by(Message.created_at.desc())
                    .limit(SUB_AGENT_CONTEXT_RECENT_LIMIT)
                )
            )
            recent = list(reversed(recent_rows))

            conv = await session.scalar(
                select(Conversation).where(Conversation.id == self.conversation_id)
            )
            pin_ids = list(conv.pinned_message_ids or []) if conv is not None else []
            pinned = (
                list(
                    await session.scalars(
                        select(Message)
                        .where(Message.id.in_(pin_ids))
                        .order_by(Message.created_at.asc())
                    )
                )
                if pin_ids
                else []
            )

            agent_ids = {m.agent_id for m in [*recent, *pinned] if m.agent_id}
            agent_name_by_id: dict[str, str] = {}
            if agent_ids:
                rows = await session.execute(
                    select(Agent.id, Agent.name).where(Agent.id.in_(agent_ids))
                )
                agent_name_by_id = {row[0]: row[1] for row in rows.all()}

        data = SubAgentPromptData(
            task=task,
            resolved_inputs=resolved_inputs,
            workspace_mode=self.workspace.mode,
            upstream_artifacts=upstream_artifacts,
            existing_artifacts=existing,
            summary_block=(
                render_conversation_summary_block(latest_summary)
                if latest_summary is not None
                else None
            ),
            recent_messages=recent,
            pinned_messages=pinned,
            agent_name_by_id=agent_name_by_id,
        )
        return build_sub_agent_prompt(data)

    def launch_attempt(
        self, task: DispatchPlanItem, prompt: str, ctx: DagContext
    ) -> tuple[str, Awaitable[DispatchTaskResult]]:
        run_id, run_task = start_run_joined(
            conversation_id=ctx.conversation_id,
            agent_id=task.agentId,
            trigger_message_id=ctx.trigger_message_id,
            parent_run_id=ctx.parent_run_id,
            parent_signal=ctx.signal,
            override_prompt=prompt,
            require_task_report=True,
        )
        return run_id, _map_run_result(run_task)

    def get_evidence(self, run_id: str) -> RunToolEvidence:
        return get_run_tool_evidence(run_id)

    def record_command_evidence(self, run_id: str, evidence: RunCommandEvidence) -> None:
        record_run_command(run_id, evidence)

    async def workspace_ready(self) -> bool:
        async with SessionLocal() as session:
            row = await session.scalar(
                select(Workspace).where(Workspace.conversation_id == self.conversation_id)
            )
            return row is not None

    def effective_cwd(self) -> str:
        return get_effective_cwd(self.workspace)

    def build_prepare_command(self, required_cwd: str | None) -> PreparedCommand | None:
        """package.json 在、node_modules 不在时补一条 `pnpm install`；其余情况不补。"""
        cwd_abs = (
            assert_path_within_workspace(self.workspace, required_cwd)
            if required_cwd
            else get_effective_cwd(self.workspace)
        )
        if not os.path.exists(os.path.join(cwd_abs, "package.json")):
            return None
        if os.path.exists(os.path.join(cwd_abs, "node_modules")):
            return None
        return PreparedCommand(command="pnpm install", cwd=required_cwd)

    async def execute_command(
        self,
        spec: PreparedCommand,
        *,
        timeout_ms: int,
        prepare: bool,
        task: DispatchPlanItem,
        run_id: str,
        ctx: DagContext,
    ) -> VerificationCommandResult:
        """验证/准备命令走 bash 工具同一条安全链（黑名单 → 路径 → 审批 → 执行）。"""
        from app.tools.bash import execute_bash_command
        from app.tools.types import ToolContext

        tool_ctx = ToolContext(
            conversation_id=ctx.conversation_id,
            agent_id=task.agentId,
            run_id=run_id,
            workspace_path="",
            abort_signal=ctx.signal,
        )
        tool_result = await execute_bash_command(
            spec.command,
            spec.cwd,
            timeout_ms,
            tool_ctx,
            evidence_kind="prepare" if prepare else "verification",
        )
        if not tool_result.ok:
            return VerificationCommandResult(
                command=spec.command,
                cwd=spec.cwd,
                ok=False,
                exit_code=None,
                timed_out=False,
                error=tool_result.error,
                prepare=prepare,
            )
        value = tool_result.value if isinstance(tool_result.value, dict) else {}
        raw_exit = value.get("exitCode")
        exit_code = raw_exit if isinstance(raw_exit, int) else None
        timed_out = value.get("timedOut") is True
        raw_command = value.get("command")
        raw_cwd = value.get("cwd")
        raw_output = value.get("output")
        return VerificationCommandResult(
            command=raw_command if isinstance(raw_command, str) else spec.command,
            cwd=raw_cwd if isinstance(raw_cwd, str) else spec.cwd,
            ok=exit_code == 0 and not timed_out,
            exit_code=exit_code,
            timed_out=timed_out,
            output=raw_output if isinstance(raw_output, str) else None,
            prepare=prepare,
        )

    async def create_project_artifact(
        self,
        task: DispatchPlanItem,
        evidence: RunToolEvidence,
        result: DispatchTaskResult,
    ) -> str | None:
        # 只负责创建；挂到 result.artifact_ids 由 run_child_task 做
        return await maybe_create_project_artifact(
            evidence_file_writes=evidence.fileWrites,
            conversation_id=self.conversation_id,
            agent_id=task.agentId,
            task_id=task.id,
        )


async def _map_run_result(run_awaitable: Awaitable[dict[str, Any]]) -> DispatchTaskResult:
    raw = await run_awaitable
    task_report = raw.get("task_report")
    return DispatchTaskResult(
        run_id=raw.get("run_id"),
        status=raw.get("status"),  # type: ignore[arg-type]
        error=raw.get("error"),
        artifact_ids=list(raw.get("artifact_ids") or []),
        output_message_ids=list(raw.get("output_message_ids") or []),
        output_artifacts=dict(raw.get("output_artifacts") or {}),
        task_report=task_report if isinstance(task_report, TaskResultReport) else None,
    )


# ─── 阶段机 ─────────────────────────────────────────────────


async def execute_orchestrator_run(
    session: AsyncSession,
    *,
    run_id: str,
    signal: AbortSignal,
    agent: Agent,
    conv: Conversation | None,
    workspace: Workspace,
    trigger: Message,
    user_prompt: str,
    attachments: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """PLAN→REVIEW→EXECUTE 循环（补救轮 ≤ MAX_DISPATCH_ROUNDS）→ AGGREGATE。

    返回执行结果（产物 id / 输出消息 id / outputKey 产物映射），run 终态由调用方收尾。
    """
    if conv is None:
        raise ValueError(f"Conversation not found: {trigger.conversation_id}")

    other_agents = await _load_other_agents(session, conv, agent.id)

    all_artifact_ids: list[str] = []
    all_output_message_ids: list[str] = []
    all_output_artifacts: dict[str, str] = {}

    def merge_execution(execution: dict[str, Any]) -> None:
        all_artifact_ids.extend(execution.get("artifact_ids") or [])
        all_output_message_ids.extend(execution.get("output_message_ids") or [])
        all_output_artifacts.update(execution.get("output_artifacts") or {})

    def current_execution() -> dict[str, Any]:
        return {
            "artifact_ids": all_artifact_ids,
            "output_message_ids": all_output_message_ids,
            "output_artifacts": all_output_artifacts,
            "task_report": None,
        }

    merged_results: dict[str, DispatchTaskResult] = {}
    plan_items_by_id: dict[str, DispatchPlanItem] = {}
    last_conflicts: list[Any] = []
    runtime = AgentChildRuntime(trigger.conversation_id, workspace)

    async def _run_child(
        task: DispatchPlanItem,
        upstream: dict[str, DispatchTaskResult],
        plan_context: list[DispatchPlanItem],
        ctx: DagContext,
    ) -> DispatchTaskResult:
        return await run_child_task(task, upstream, plan_context, ctx, runtime)

    for round_index in range(1, MAX_DISPATCH_ROUNDS + 1):
        if signal.aborted:
            raise RuntimeError("Orchestrator run aborted")

        # 补救轮把上一轮的失败/冲突摘要喂回 plan 阶段，原始请求保留在 <original_request>
        replan_context = (
            None
            if round_index == 1
            else build_replan_context(
                to_replan_views(plan_items_by_id, merged_results),
                to_replan_conflicts(last_conflicts),
            )
        )

        initial_plan, plan_execution = await run_plan_stage(
            session,
            run_id=run_id,
            signal=signal,
            agent=agent,
            conv=conv,
            workspace=workspace,
            trigger=trigger,
            user_prompt=user_prompt,
            other_agents=other_agents,
            attachments=attachments if round_index == 1 else None,
            replan_context=replan_context,
            resolved_external_tasks=list(plan_items_by_id.values()),
        )
        merge_execution(plan_execution)

        if initial_plan is None:
            # 首轮没拆 plan = 直接回答了用户；补救轮 = 判断无需/无法补救，进聚合
            if round_index == 1:
                return current_execution()
            break

        # ─── REVIEW：批准 / 拒绝 / 修改（修改 → 重排后再审）───
        plan = initial_plan
        approved_plan: list[DispatchPlanItem] | None = None
        while True:
            outcome = await wait_for_dispatch_plan_review(
                conversation_id=trigger.conversation_id,
                agent_id=agent.id,
                run_id=run_id,
                plan=plan,
                available_agents=other_agents,
                orchestrator_agent_id=agent.id,
                resolved_external_tasks=list(plan_items_by_id.values()),
                signal=signal,
            )
            if outcome.get("kind") == "approve":
                approved_plan = outcome.get("plan") or plan
                break
            if outcome.get("kind") == "reject":
                break

            # revise：反馈喂回 plan 阶段重排，新计划继续进审批门禁
            revised_plan, revised_execution = await run_plan_stage(
                session,
                run_id=run_id,
                signal=signal,
                agent=agent,
                conv=conv,
                workspace=workspace,
                trigger=trigger,
                user_prompt=user_prompt,
                other_agents=other_agents,
                attachments=None,
                replan_context=build_revise_context(plan, outcome.get("feedback", "")),
                resolved_external_tasks=list(plan_items_by_id.values()),
            )
            merge_execution(revised_execution)
            if revised_plan is not None:
                plan = revised_plan

        if approved_plan is None:
            if signal.aborted:
                raise RuntimeError("Orchestrator run aborted")
            # 用户拒绝：首轮直接结束；补救轮用已有结果进聚合
            if round_index == 1:
                return current_execution()
            break

        event_bus.publish(
            DispatchPlanEvent(
                conversationId=trigger.conversation_id,
                timestamp=now_ms(),
                runId=run_id,
                plan=approved_plan,
            )
        )
        for item in approved_plan:
            plan_items_by_id[item.id] = item

        # ─── EXECUTE：按依赖波次并发跑 ───────────────────────
        results, conflicts = await execute_dag(
            approved_plan,
            DagContext(
                parent_run_id=run_id,
                conversation_id=trigger.conversation_id,
                trigger_message_id=trigger.id,
                signal=signal,
                seed_results=dict(merged_results),
                external_plan_items=list(plan_items_by_id.values()),
            ),
            _run_child,
        )
        merged_results.update(results)
        last_conflicts = conflicts

        round_views = [
            ReplanTaskView(
                taskId=item.id,
                agentId=item.agentId,
                status=(
                    results[item.id].status
                    if item.id in results and results[item.id].status
                    else "skipped"
                ),
                error=results[item.id].error if item.id in results else None,
            )
            for item in approved_plan
        ]
        if not should_replan(round_views, to_replan_conflicts(conflicts)):
            break

    # 合并后的最终产物（按任务合并，补救轮旧结果不重复登记 —— execute_dag 只回本轮任务）
    for result in merged_results.values():
        all_artifact_ids.extend(result.artifact_ids)
        all_output_message_ids.extend(result.output_message_ids)
        all_output_artifacts.update(result.output_artifacts)

    # ─── AGGREGATE：结果 XML → 最终总结发言 ────────────────────
    artifact_by_id = await _load_artifacts_by_id(session, all_artifact_ids)
    aggregate_system_prompt = build_orchestrator_aggregate_prompt(agent.system_prompt)
    aggregate_user_prompt = build_aggregate_prompt(
        user_prompt,
        list(plan_items_by_id.values()),
        merged_results,
        last_conflicts,
        get_effective_cwd(workspace),
        artifact_by_id,
    )
    # 聚合阶段不再带 plan_tasks / ask_user：不重复拆解，也不在总结前再打断用户
    aggregate_tool_names = [
        name
        for name in (agent.tool_names or [])
        if name != "plan_tasks" and name != ASK_USER_TOOL_NAME
    ]
    aggregate_input = await build_adapter_input(
        session,
        agent=agent,
        conv=conv,
        workspace=workspace,
        run_id=run_id,
        prompt=aggregate_user_prompt,
        tool_names=aggregate_tool_names,
        system_prompt_override=aggregate_system_prompt,
        # 不再带原始附件：plan 阶段已经看过，重复传图浪费 token
        attachments=None,
        conversation_id=trigger.conversation_id,
        exclude_message_id=trigger.id,
        include_history=True,
    )
    adapter = get_adapter(agent.adapter_name)
    aggregate_execution = await consume_stream(session, adapter, aggregate_input, signal, run_id)
    merge_execution(aggregate_execution)

    return current_execution()


async def run_plan_stage(
    session: AsyncSession,
    *,
    run_id: str,
    signal: AbortSignal,
    agent: Agent,
    conv: Conversation | None,
    workspace: Workspace,
    trigger: Message,
    user_prompt: str,
    other_agents: list[Agent],
    attachments: list[dict[str, Any]] | None,
    replan_context: str | None,
    resolved_external_tasks: list[DispatchPlanItem],
) -> tuple[list[DispatchPlanItem] | None, dict[str, Any]]:
    """一轮 plan 阶段：plan_tasks 一到就停流，再编译校验出可执行计划。

    没调 plan_tasks（直接答复 / 判定无需补救）时返回 (None, 执行结果)。
    """
    plan_system_prompt = build_orchestrator_plan_prompt(
        agent.system_prompt, other_agents, workspace.mode
    )
    plan_tool_names = _ensure_includes(
        _ensure_includes(
            [
                name
                for name in (agent.tool_names or [])
                if name in ORCHESTRATOR_PLAN_ALLOWED_TOOLS
            ],
            "plan_tasks",
        ),
        ASK_USER_TOOL_NAME,
    )
    # 补救/重排轮：上下文摘要在前，原始请求保留在 <original_request>
    effective_prompt = (
        f"{replan_context}\n\n<original_request>\n{user_prompt}\n</original_request>"
        if replan_context
        else user_prompt
    )

    plan_ref: dict[str, Any] = {"value": None}

    def on_tool_call(event: ToolCallEvent) -> dict[str, Any] | None:
        plan_args = extract_plan_tasks_tool_args(
            {"toolName": event.toolName, "args": event.args}
        )
        if plan_args is None:
            return None
        plan = parse_dispatch_plan_tool_args(plan_args)
        plan_ref["value"] = plan
        return {"stop": True, "result": {"acknowledged": True, "taskCount": len(plan)}}

    plan_input = await build_adapter_input(
        session,
        agent=agent,
        conv=conv,
        workspace=workspace,
        run_id=run_id,
        prompt=effective_prompt,
        tool_names=plan_tool_names,
        system_prompt_override=plan_system_prompt,
        attachments=attachments,
        conversation_id=trigger.conversation_id,
        exclude_message_id=trigger.id,
        include_history=True,
    )
    adapter = get_adapter(agent.adapter_name)
    plan_execution = await consume_stream(
        session, adapter, plan_input, signal, run_id, on_tool_call=on_tool_call
    )

    raw_plan: list[DispatchPlanItem] | None = plan_ref["value"]
    plan = (
        compile_and_validate_dispatch_plan(
            raw_plan, other_agents, agent.id, resolved_external_tasks
        )
        if raw_plan is not None
        else None
    )
    return plan, plan_execution


async def wait_for_dispatch_plan_review(
    *,
    conversation_id: str,
    agent_id: str,
    run_id: str,
    plan: list[DispatchPlanItem],
    available_agents: list[Any],
    orchestrator_agent_id: str,
    resolved_external_tasks: list[DispatchPlanItem] | None = None,
    signal: AbortSignal,
) -> PlanReviewOutcome:
    """登记待审计划并挂起等用户决定；abort 走拒绝收口（前端收到 resolved 关卡片）。"""

    def validator(candidate: list[DispatchPlanItem]) -> list[DispatchPlanItem]:
        return compile_and_validate_dispatch_plan(
            candidate, available_agents, orchestrator_agent_id, resolved_external_tasks
        )

    pending = pending_dispatch_plans.register(
        conversation_id=conversation_id,
        agent_id=agent_id,
        run_id=run_id,
        plan=plan,
        validator=validator,
    )

    loop = asyncio.get_running_loop()
    future: asyncio.Future[PlanReviewOutcome] = loop.create_future()

    def resolve(outcome: PlanReviewOutcome) -> None:
        if not future.done():
            future.set_result(outcome)

    if not pending_dispatch_plans.attach_resolver(pending.id, resolve):
        # 表项已经不在了（理论上只发生在取消竞态）：按拒绝收口
        return {"kind": "reject"}

    def on_abort() -> None:
        pending_dispatch_plans.cancel(pending.id)

    signal.add_listener(on_abort)
    try:
        return await future
    finally:
        signal.remove_listener(on_abort)


async def _load_other_agents(
    session: AsyncSession, conv: Conversation, orchestrator_agent_id: str
) -> list[Agent]:
    other_ids = [agent_id for agent_id in (conv.agent_ids or []) if agent_id != orchestrator_agent_id]
    if not other_ids:
        return []
    return list(await session.scalars(select(Agent).where(Agent.id.in_(other_ids))))


async def _load_artifacts_by_id(
    session: AsyncSession, artifact_ids: list[str]
) -> dict[str, Artifact]:
    unique_ids = list(dict.fromkeys(artifact_ids))
    if not unique_ids:
        return {}
    rows = list(await session.scalars(select(Artifact).where(Artifact.id.in_(unique_ids))))
    return {row.id: row for row in rows}
