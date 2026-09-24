"""DAG 调度器：把批准的分派计划按依赖波次并发执行，产出每个子任务的终态。

调度语义（行为契约）：
- 顺序只认 dependsOn：所有依赖 complete 的任务构成「一波」，波内并发跑；
- 依赖 failed / aborted / skipped → 本任务 skipped（失败传播，独立分支照常跑完）；
- 上游没交出 required input 产物 → 本任务 skipped；
- 中止信号一到：没起跑的任务标 aborted，跑着的子 run 经 parent signal 级联停下；
- 同波次 ≥2 个子 run 以不同内容写了同一文件 → 计入 conflicts（只上报，不合并）；
- 每个子任务恰好发一条 dispatch.end；子 run 起跑时发 dispatch.start。

子 run 怎么起、prompt 怎么拼、验证命令怎么执行，走 ChildTaskRuntime 注入 ——
调度器只管编排，固定计划可以直接喂进来验证，不经过 LLM。写入/命令证据
按 run 隔离（dispatch_file_writes / dispatch_run_evidence），波次间互不污染。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Protocol

from app.schemas.dispatch import (
    DispatchPlanItem,
    DispatchRequiredCommand,
    DispatchTaskEndStatus,
    DispatchTaskInput,
    RunCommandEvidence,
    RunToolEvidence,
    TaskResultReport,
)
from app.schemas.events import DispatchEndEvent, DispatchStartEvent
from app.services.dispatch_file_writes import (
    FileWriteConflict,
    RunFileWrites,
    clear_file_writes,
    detect_wave_conflicts,
    get_file_writes,
)
from app.services.dispatch_plan import get_required_expected_outputs
from app.services.dispatch_run_evidence import clear_run_tool_evidence
from app.services.event_bus import event_bus
from app.services.task_result_report import evaluate_task_result_report
from app.utils.abort import AbortSignal
from app.utils.time import now_ms

# 子任务重试上限：首轮 + 补救（失败时把上下文回灌给子 agent 再试）
MAX_CHILD_TASK_ATTEMPTS = 4
DEFAULT_VERIFICATION_TIMEOUT_MS = 5 * 60_000
DEFAULT_PREPARE_TIMEOUT_MS = 10 * 60_000

# requiredCommands 里出现包安装时不再自动补 install 准备命令（否则重复装）
_INSTALL_COMMAND_PATTERN = re.compile(r"\b(?:pnpm|npm|yarn)\s+install\b", re.IGNORECASE)
_CD_PREFIX_PATTERN = re.compile(r'^cd\s+("?[^"&;]+"?)\s*&&\s*(.+)$', re.IGNORECASE)


class Semaphore:
    """带 abort 感知的并发闸门：排队中被中止立刻失败，不占额度。"""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._active = 0
        self._queue: list[tuple[asyncio.Future, AbortSignal, Callable[[], None]]] = []

    async def acquire(self, signal: AbortSignal) -> Callable[[], None]:
        if signal.aborted:
            raise RuntimeError("Semaphore acquire aborted")
        if self._active < self._limit:
            self._active += 1
            return self._create_release()

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future = loop.create_future()

        def on_abort() -> None:
            for item in list(self._queue):
                if item[0] is waiter:
                    self._queue.remove(item)
                    break
            if not waiter.done():
                waiter.set_exception(RuntimeError("Semaphore acquire aborted"))
                # 异常没人 await 也不留告警
                waiter.exception()

        signal.add_listener(on_abort)
        self._queue.append((waiter, signal, on_abort))
        try:
            return await waiter
        finally:
            signal.remove_listener(on_abort)

    def _create_release(self) -> Callable[[], None]:
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            self._active -= 1
            self._drain()

        return release

    def _drain(self) -> None:
        while self._active < self._limit and self._queue:
            waiter, signal, on_abort = self._queue.pop(0)
            signal.remove_listener(on_abort)
            if waiter.done():
                continue
            self._active += 1
            waiter.set_result(self._create_release())


@dataclass
class DispatchTaskResult:
    """一次子任务（或其单次尝试）的执行结果。status 为空表示聚合中的中间态。"""

    run_id: str | None = None
    run_ids: list[str] | None = None
    status: DispatchTaskEndStatus | None = None
    error: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    output_message_ids: list[str] = field(default_factory=list)
    output_artifacts: dict[str, str] = field(default_factory=dict)
    task_report: TaskResultReport | None = None


@dataclass
class ResolvedTaskInput:
    """一条 inputs 引用的解析结果：产物 id（缺了就是 missing）与上游契约里的类型。"""

    input: DispatchTaskInput
    type: str | None
    artifact_id: str | None
    missing: bool


@dataclass
class BlockedDependency:
    task_id: str
    result: DispatchTaskResult


@dataclass
class VerificationCommandResult:
    """一条验证/准备命令的执行结果（补跑 requiredCommands 的观察记录）。"""

    command: str
    ok: bool
    exit_code: int | None
    timed_out: bool
    cwd: str | None = None
    output: str | None = None
    error: str | None = None
    prepare: bool = False


@dataclass
class PreparedCommand:
    command: str
    cwd: str | None = None


@dataclass
class ChildAttemptEvaluation:
    raw_result: DispatchTaskResult
    result: DispatchTaskResult
    evidence: RunToolEvidence
    verification_results: list[VerificationCommandResult]


@dataclass
class DagContext:
    """一轮 DAG 执行的上下文；seed_results 携带补救轮已有的上游结果。"""

    parent_run_id: str
    conversation_id: str
    trigger_message_id: str
    signal: AbortSignal
    seed_results: dict[str, DispatchTaskResult] | None = None
    external_plan_items: list[DispatchPlanItem] | None = None
    semaphore: Semaphore | None = None


class ChildTaskRuntime(Protocol):
    """子任务执行的对外依赖面（生产实现接适配器/DB，测试喂假实现）。"""

    async def build_sub_agent_prompt(
        self,
        task: DispatchPlanItem,
        resolved_inputs: list[ResolvedTaskInput],
        upstream: dict[str, DispatchTaskResult],
        plan: list[DispatchPlanItem],
    ) -> str: ...

    def launch_attempt(
        self, task: DispatchPlanItem, prompt: str, ctx: DagContext
    ) -> tuple[str, Awaitable[DispatchTaskResult]]: ...

    def get_evidence(self, run_id: str) -> RunToolEvidence: ...

    def record_command_evidence(self, run_id: str, evidence: RunCommandEvidence) -> None: ...

    async def workspace_ready(self) -> bool: ...

    def effective_cwd(self) -> str: ...

    def build_prepare_command(self, required_cwd: str | None) -> PreparedCommand | None: ...

    async def execute_command(
        self,
        spec: PreparedCommand,
        *,
        timeout_ms: int,
        prepare: bool,
        task: DispatchPlanItem,
        run_id: str,
        ctx: DagContext,
    ) -> VerificationCommandResult: ...

    async def create_project_artifact(
        self,
        task: DispatchPlanItem,
        evidence: RunToolEvidence,
        result: DispatchTaskResult,
    ) -> str | None: ...


RunChildFn = Callable[
    [DispatchPlanItem, dict[str, DispatchTaskResult], list[DispatchPlanItem], DagContext],
    Awaitable[DispatchTaskResult],
]

# 全局并发上限：跨 run 限制同时在跑的子 agent 数
_sub_agent_semaphore = Semaphore(4)


# ─── 结果/证据的合并工具 ────────────────────────────────────
def get_dispatch_result_run_ids(result: DispatchTaskResult) -> list[str]:
    if result.run_ids:
        return list(result.run_ids)
    return [result.run_id] if result.run_id else []


def merge_file_writes(run_ids: list[str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for run_id in run_ids:
        merged.update(get_file_writes(run_id))
    return merged


def merge_run_execution_result(target: DispatchTaskResult, source: DispatchTaskResult) -> None:
    target.artifact_ids.extend(source.artifact_ids)
    target.output_message_ids.extend(source.output_message_ids)
    target.output_artifacts.update(source.output_artifacts)
    if source.task_report is not None:
        target.task_report = source.task_report


def merge_run_tool_evidence(target: RunToolEvidence, source: RunToolEvidence) -> None:
    target.fileWrites.extend(source.fileWrites)
    target.commands.extend(source.commands)


def clone_run_tool_evidence(source: RunToolEvidence) -> RunToolEvidence:
    return RunToolEvidence(fileWrites=list(source.fileWrites), commands=list(source.commands))


def merge_attempt_aggregate(
    result: DispatchTaskResult, aggregate: DispatchTaskResult
) -> DispatchTaskResult:
    return replace(
        result,
        run_ids=result.run_ids,
        artifact_ids=merge_unique([*aggregate.artifact_ids, *result.artifact_ids]),
        output_message_ids=list(aggregate.output_message_ids),
        output_artifacts={**aggregate.output_artifacts, **result.output_artifacts},
    )


def merge_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


# ─── 纯函数：输入解析 / 产物绑定 / 终态构造 ─────────────────
def resolve_task_inputs(
    task: DispatchPlanItem,
    upstream: dict[str, DispatchTaskResult],
    plan: list[DispatchPlanItem],
) -> list[ResolvedTaskInput]:
    task_by_id = {item.id: item for item in plan}
    resolved: list[ResolvedTaskInput] = []
    for input_ in task.inputs or []:
        upstream_task = task_by_id.get(input_.fromTaskId)
        expected_output = None
        for output in upstream_task.expectedOutputs or []:
            if output.id == input_.outputId:
                expected_output = output
                break
        upstream_result = upstream.get(input_.fromTaskId)
        artifact_id = (
            upstream_result.output_artifacts.get(input_.outputId)
            if upstream_result is not None
            else None
        )
        resolved.append(
            ResolvedTaskInput(
                input=input_,
                type=expected_output.type if expected_output is not None else None,
                artifact_id=artifact_id,
                missing=not artifact_id,
            )
        )
    return resolved


def bind_implicit_single_output(
    task: DispatchPlanItem, result: DispatchTaskResult
) -> dict[str, str]:
    """单产物任务：只产出一件 artifact 时把它绑到唯一的 required output 上。"""
    output_artifacts = dict(result.output_artifacts)
    required_outputs = get_required_expected_outputs(task)
    if len(required_outputs) != 1 or len(result.artifact_ids) != 1:
        return output_artifacts

    output_id = required_outputs[0].id
    if output_artifacts.get(output_id):
        return output_artifacts
    if result.artifact_ids[0] in output_artifacts.values():
        return output_artifacts
    output_artifacts[output_id] = result.artifact_ids[0]
    return output_artifacts


def bind_project_expected_output(
    task: DispatchPlanItem,
    result: DispatchTaskResult,
    project_artifact_id: str | None,
) -> DispatchTaskResult:
    if not project_artifact_id:
        return result
    project_outputs = [
        output for output in get_required_expected_outputs(task) if output.type == "project"
    ]
    if not project_outputs:
        return result

    output_artifacts = dict(result.output_artifacts)
    for output in project_outputs:
        output_artifacts.setdefault(output.id, project_artifact_id)
    return replace(result, output_artifacts=output_artifacts)


def evaluate_required_project_outputs(
    task: DispatchPlanItem, result: DispatchTaskResult
) -> dict[str, Any]:
    missing = [
        output
        for output in get_required_expected_outputs(task)
        if output.type == "project" and output.id not in result.output_artifacts
    ]
    if not missing:
        return {"ok": True}
    return {
        "ok": False,
        "error": 'Task "{}" is missing required project output: {}'.format(
            task.id, ", ".join(output.id for output in missing)
        ),
    }


def evaluate_child_task_result(
    task: DispatchPlanItem,
    result: DispatchTaskResult,
    evidence: RunToolEvidence | None = None,
) -> DispatchTaskResult:
    """完成门禁：run 自称 complete 也要过证据判定，缺证据翻成 failed。"""
    if result.status != "complete":
        return result

    output_artifacts = bind_implicit_single_output(task, result)
    report_evaluation = evaluate_task_result_report(
        task, result.task_report, evidence if evidence is not None else RunToolEvidence()
    )
    if not report_evaluation["ok"]:
        return replace(
            result,
            output_artifacts=output_artifacts,
            status="failed",
            error=report_evaluation["error"],
        )

    return replace(result, output_artifacts=output_artifacts)


def skipped_missing_inputs_task_result(
    task: DispatchPlanItem, missing_inputs: list[ResolvedTaskInput]
) -> DispatchTaskResult:
    missing_text = ", ".join(
        f"{entry.input.fromTaskId}.{entry.input.outputId}" for entry in missing_inputs
    )
    return DispatchTaskResult(
        run_id=None,
        status="skipped",
        error=(
            f'Skipped because required input artifact(s) were missing for task "{task.id}": '
            f"{missing_text}"
        ),
    )


def skipped_task_result(
    task: DispatchPlanItem, blockers: list[BlockedDependency]
) -> DispatchTaskResult:
    blocker_text = ", ".join(
        f"{blocker.task_id}:{blocker.result.status}" for blocker in blockers
    )
    return DispatchTaskResult(
        run_id=None,
        status="skipped",
        error=(
            f'Skipped because upstream task(s) did not complete for task "{task.id}": '
            f"{blocker_text}"
        ),
    )


def aborted_before_start_task_result(task: DispatchPlanItem, error: str) -> DispatchTaskResult:
    return DispatchTaskResult(run_id=None, status="aborted", error=f'{error} for task "{task.id}"')


def mark_remaining_tasks_aborted(
    plan: list[DispatchPlanItem],
    remaining: set[str],
    results: dict[str, DispatchTaskResult],
    ctx: DagContext,
) -> None:
    for task in plan:
        if task.id not in remaining:
            continue
        result = DispatchTaskResult(
            run_id=None,
            status="aborted",
            error=f'Aborted before task "{task.id}" started',
        )
        results[task.id] = result
        remaining.discard(task.id)
        publish_dispatch_end(ctx, task.id, result)


def publish_dispatch_end(ctx: DagContext, task_id: str, result: DispatchTaskResult) -> None:
    event_bus.publish(
        DispatchEndEvent(
            conversationId=ctx.conversation_id,
            timestamp=now_ms(),
            parentRunId=ctx.parent_run_id,
            childRunId=result.run_id,
            taskId=task_id,
            status=result.status or "aborted",
            error=result.error,
        )
    )


# ─── requiredCommands 补跑 ─────────────────────────────────
@dataclass
class _ExpandedCommand:
    cwd: str | None
    commands: list[str]


def expand_required_command(required: DispatchRequiredCommand) -> _ExpandedCommand:
    """拆 `cd xxx && a && b` 形态：cd 前缀变 cwd，`&&` 两侧拆成顺序命令。"""
    cwd = required.cwd
    command = required.command.strip()
    cd_match = _CD_PREFIX_PATTERN.match(command)
    if cd_match and not cwd:
        cwd = cd_match.group(1).strip('"')
        command = cd_match.group(2).strip()
    commands = [part.strip() for part in re.split(r"\s+&&\s+", command) if part.strip()]
    return _ExpandedCommand(cwd=cwd, commands=commands)


async def run_required_commands(
    task: DispatchPlanItem,
    run_id: str,
    ctx: DagContext,
    runtime: ChildTaskRuntime,
) -> list[VerificationCommandResult]:
    """任务契约里的 requiredCommands 逐条补跑；一条失败就停这条，继续下一条。"""
    if not task.requiredCommands:
        return []

    if not await runtime.workspace_ready():
        return [
            VerificationCommandResult(
                command="(required commands)",
                ok=False,
                exit_code=None,
                timed_out=False,
                error="Workspace not found",
            )
        ]

    results: list[VerificationCommandResult] = []
    for required in task.requiredCommands:
        expanded = expand_required_command(required)
        command_results: list[VerificationCommandResult] = []

        prepare: PreparedCommand | None
        try:
            needs_no_prepare = any(
                _INSTALL_COMMAND_PATTERN.search(command) for command in expanded.commands
            )
            prepare = (
                None if needs_no_prepare else runtime.build_prepare_command(expanded.cwd)
            )
        except Exception as err:  # noqa: BLE001 - 准备阶段的任何错误转成一条失败结果
            results.append(
                VerificationCommandResult(
                    command="prepare workspace",
                    ok=False,
                    exit_code=None,
                    timed_out=False,
                    cwd=expanded.cwd,
                    error=str(err),
                    prepare=True,
                )
            )
            continue

        if prepare is not None:
            prepare_result = await runtime.execute_command(
                prepare,
                timeout_ms=DEFAULT_PREPARE_TIMEOUT_MS,
                prepare=True,
                task=task,
                run_id=run_id,
                ctx=ctx,
            )
            results.append(prepare_result)
            command_results.append(prepare_result)
            if not prepare_result.ok:
                continue

        for command in expanded.commands:
            command_result = await runtime.execute_command(
                PreparedCommand(command=command, cwd=expanded.cwd),
                timeout_ms=required.timeoutMs or DEFAULT_VERIFICATION_TIMEOUT_MS,
                prepare=False,
                task=task,
                run_id=run_id,
                ctx=ctx,
            )
            results.append(command_result)
            command_results.append(command_result)
            if not command_result.ok:
                break

        failed = next((item for item in command_results if not item.ok), None)
        first_cwd = command_results[0].cwd if command_results else None
        cwd = first_cwd if first_cwd is not None else required.cwd
        runtime.record_command_evidence(
            run_id,
            RunCommandEvidence(
                command=required.command,
                cwd=cwd if cwd is not None else runtime.effective_cwd(),
                exitCode=failed.exit_code if failed is not None else 0,
                timedOut=bool(failed and failed.timed_out),
                isError=failed is not None,
                error=failed.error if failed is not None and failed.error else None,
            ),
        )

    return results


# ─── 子任务执行（尝试循环 + 续跑上下文）────────────────────
async def run_child_task_attempt(
    task: DispatchPlanItem,
    prompt: str,
    ctx: DagContext,
    runtime: ChildTaskRuntime,
) -> ChildAttemptEvaluation:
    child_run_id, raw_awaitable = runtime.launch_attempt(task, prompt, ctx)
    event_bus.publish(
        DispatchStartEvent(
            conversationId=ctx.conversation_id,
            timestamp=now_ms(),
            parentRunId=ctx.parent_run_id,
            childRunId=child_run_id,
            taskId=task.id,
            agentId=task.agentId,
        )
    )
    raw = await raw_awaitable
    verification_results = (
        []
        if raw.status == "aborted"
        else await run_required_commands(task, child_run_id, ctx, runtime)
    )
    evidence = runtime.get_evidence(child_run_id)
    result = evaluate_child_task_result(task, raw, evidence)
    return ChildAttemptEvaluation(
        raw_result=raw,
        result=result,
        evidence=evidence,
        verification_results=verification_results,
    )


async def run_child_task(
    task: DispatchPlanItem,
    upstream: dict[str, DispatchTaskResult],
    plan: list[DispatchPlanItem],
    ctx: DagContext,
    runtime: ChildTaskRuntime,
) -> DispatchTaskResult:
    """跑一个子任务：缺输入先 skip，并发闸门排队，失败带上下文续跑最多 N 次。"""
    resolved_inputs = resolve_task_inputs(task, upstream, plan)
    missing_required_inputs = [
        entry for entry in resolved_inputs if entry.missing and entry.input.required is not False
    ]
    if missing_required_inputs:
        return skipped_missing_inputs_task_result(task, missing_required_inputs)

    semaphore = ctx.semaphore or _sub_agent_semaphore
    try:
        release = await semaphore.acquire(ctx.signal)
    except RuntimeError:
        return aborted_before_start_task_result(
            task, "Aborted while waiting for sub-agent concurrency slot"
        )

    try:
        base_prompt = await runtime.build_sub_agent_prompt(task, resolved_inputs, upstream, plan)

        continuation_context: str | None = None
        last_evaluation: ChildAttemptEvaluation | None = None
        aggregate = DispatchTaskResult()
        aggregate_evidence = RunToolEvidence()
        attempt_run_ids: list[str] = []

        for attempt in range(1, MAX_CHILD_TASK_ATTEMPTS + 1):
            if ctx.signal.aborted:
                return merge_attempt_aggregate(
                    aborted_before_start_task_result(task, "Aborted before sub-agent run started"),
                    aggregate,
                )

            prompt = (
                build_continuation_prompt(base_prompt, task, attempt, continuation_context)
                if continuation_context is not None
                else base_prompt
            )
            attempt_evaluation = await run_child_task_attempt(task, prompt, ctx, runtime)
            if attempt_evaluation.raw_result.run_id:
                attempt_run_ids.append(attempt_evaluation.raw_result.run_id)
            merge_run_execution_result(aggregate, attempt_evaluation.raw_result)
            merge_run_tool_evidence(aggregate_evidence, attempt_evaluation.evidence)

            # 用累计证据重新过门禁：多轮补跑的证据能补上前几轮的缺口
            evaluated_result = evaluate_child_task_result(
                task, attempt_evaluation.raw_result, aggregate_evidence
            )
            current = replace(
                attempt_evaluation,
                result=replace(evaluated_result, run_ids=list(attempt_run_ids)),
                evidence=clone_run_tool_evidence(aggregate_evidence),
            )

            if current.result.status == "complete":
                project_artifact_id = await runtime.create_project_artifact(
                    task, aggregate_evidence, current.result
                )
                if project_artifact_id:
                    current.result.artifact_ids.append(project_artifact_id)
                result_with_project = bind_project_expected_output(
                    task, current.result, project_artifact_id
                )
                output_evaluation = evaluate_required_project_outputs(task, result_with_project)
                current = replace(
                    current,
                    result=result_with_project
                    if output_evaluation["ok"]
                    else replace(
                        result_with_project, status="failed", error=output_evaluation["error"]
                    ),
                )

            last_evaluation = current

            if current.result.status == "complete":
                return merge_attempt_aggregate(current.result, aggregate)

            if current.result.status == "aborted" or (
                current.result.task_report is not None
                and current.result.task_report.status == "blocked"
            ):
                return merge_attempt_aggregate(current.result, aggregate)

            continuation_context = build_task_continuation_context(
                task, current, attempt, MAX_CHILD_TASK_ATTEMPTS
            )

        result = (
            last_evaluation.result
            if last_evaluation is not None
            else aborted_before_start_task_result(task, "No child task attempt was executed")
        )
        exhausted = replace(
            result,
            status="complete" if result.status == "complete" else "failed",
            error=(
                result.error
                if result.status == "complete"
                else (
                    f'Task "{task.id}" did not satisfy completion gates after '
                    f"{MAX_CHILD_TASK_ATTEMPTS} attempt(s). "
                    f"Last error: {result.error or 'unknown error'}"
                )
            ),
            run_ids=list(attempt_run_ids),
        )
        return merge_attempt_aggregate(exhausted, aggregate)
    finally:
        release()


def build_continuation_prompt(
    base_prompt: str, task: DispatchPlanItem, attempt: int, continuation_context: str
) -> str:
    return "\n".join(
        [
            base_prompt,
            "",
            "<continuation>",
            f'You are continuing the same dispatched task "{task.id}". '
            f"This is attempt {attempt}/{MAX_CHILD_TASK_ATTEMPTS}.",
            "Do not restart from scratch if useful files already exist. Inspect the workspace, "
            "fix the missing or failing parts, run the relevant verification, and then call "
            "report_task_result.",
            continuation_context,
            "</continuation>",
        ]
    )


def build_task_continuation_context(
    task: DispatchPlanItem,
    evaluation: ChildAttemptEvaluation,
    attempt: int,
    max_attempts: int,
) -> str:
    lines = [
        "<previous_attempt>",
        f"  <attempt>{attempt}/{max_attempts}</attempt>",
        f"  <status>{evaluation.result.status}</status>",
    ]
    if evaluation.result.error:
        lines.append(f"  <error>{_escape_xml(evaluation.result.error)}</error>")
    if evaluation.result.task_report is None:
        lines.append("  <missing_report>true</missing_report>")
    if task.targetPaths:
        lines.append("  <target_paths>")
        for target_path in task.targetPaths:
            lines.append(f"    <path>{_escape_xml(target_path)}</path>")
        lines.append("  </target_paths>")
    if evaluation.verification_results:
        lines.append("  <verification_results>")
        for result in evaluation.verification_results:
            attrs = (
                f"text={json.dumps(result.command, ensure_ascii=False)} "
                f'ok="{str(result.ok).lower()}" '
                f'exitCode="{result.exit_code if result.exit_code is not None else ""}" '
                f'timedOut="{str(result.timed_out).lower()}"'
            )
            if result.prepare:
                attrs += ' prepare="true"'
            lines.append(f"    <command {attrs}>")
            if result.cwd:
                lines.append(f"      <cwd>{_escape_xml(result.cwd)}</cwd>")
            if result.error:
                lines.append(f"      <error>{_escape_xml(result.error)}</error>")
            if result.output:
                lines.append(f"      <output>{_escape_xml(result.output[-4000:])}</output>")
            lines.append("    </command>")
        lines.append("  </verification_results>")
    lines.append("</previous_attempt>")
    return "\n".join(lines)


def _escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ─── 波次调度主循环 ────────────────────────────────────────
def _merge_external_plan_items(
    external_items: list[DispatchPlanItem], current_plan: list[DispatchPlanItem]
) -> list[DispatchPlanItem]:
    by_id: dict[str, DispatchPlanItem] = {}
    for item in external_items:
        by_id[item.id] = item
    for item in current_plan:
        by_id[item.id] = item
    return list(by_id.values())


async def execute_dag(
    plan: list[DispatchPlanItem],
    ctx: DagContext,
    run_child: RunChildFn,
) -> tuple[dict[str, DispatchTaskResult], list[FileWriteConflict]]:
    """按波次执行计划：ready 的任务并发跑，失败向下游传播 skipped，波尾做写冲突检测。

    返回 (本轮任务 id → 终态, 本计划内检出的写冲突)。dispatch.end 在这里统一发出，
    每个任务恰好一条；补救轮的旧结果经 seed_results 带入（不重跑、不重复上报）。
    """
    current_task_ids = {task.id for task in plan}
    results: dict[str, DispatchTaskResult] = {
        task_id: result
        for task_id, result in (ctx.seed_results or {}).items()
        if task_id not in current_task_ids
    }
    remaining = set(current_task_ids)
    conflicts: list[FileWriteConflict] = []
    plan_context = _merge_external_plan_items(ctx.external_plan_items or [], plan)

    async def _run_one(task: DispatchPlanItem) -> DispatchTaskResult:
        # 每个子任务跑完立刻发 end（不等整波结束，前端进度才逐条走）
        result = await run_child(task, results, plan_context, ctx)
        publish_dispatch_end(ctx, task.id, result)
        return result

    while remaining:
        if ctx.signal.aborted:
            mark_remaining_tasks_aborted(plan, remaining, results, ctx)
            raise RuntimeError("Orchestrator run aborted")

        for task in plan:
            if task.id not in remaining:
                continue
            blockers = [
                BlockedDependency(task_id=dep, result=results[dep])
                for dep in task.dependsOn or []
                if dep in results and results[dep].status != "complete"
            ]
            if not blockers:
                continue

            result = skipped_task_result(task, blockers)
            results[task.id] = result
            remaining.discard(task.id)
            publish_dispatch_end(ctx, task.id, result)

        if not remaining:
            break

        ready = [
            task
            for task in plan
            if task.id in remaining
            and all(results.get(dep) is not None and results[dep].status == "complete"
                    for dep in task.dependsOn or [])
        ]
        if not ready:
            raise RuntimeError("Circular dependency or unresolved task in plan")

        wave = await asyncio.gather(*[_run_one(task) for task in ready])
        for index, task in enumerate(ready):
            results[task.id] = wave[index]
            remaining.discard(task.id)

        # 同波次写冲突检测：≥2 个子 run 经 fs_write 写了同一文件且内容不同
        if len(ready) > 1:
            run_writes: list[RunFileWrites] = []
            for index, task in enumerate(ready):
                child_run_ids = get_dispatch_result_run_ids(wave[index])
                if not child_run_ids:
                    continue
                run_writes.append(
                    RunFileWrites(
                        taskId=task.id,
                        agentId=task.agentId,
                        runId=child_run_ids[-1],
                        writes=merge_file_writes(child_run_ids),
                    )
                )
            conflicts.extend(detect_wave_conflicts(run_writes))

    # 释放本次 dispatch 各子 run 的写入/证据记录（内存）
    for task_id, result in results.items():
        if task_id not in current_task_ids:
            continue
        for child_run_id in get_dispatch_result_run_ids(result):
            clear_file_writes(child_run_id)
            clear_run_tool_evidence(child_run_id)

    return (
        {
            task_id: result
            for task_id, result in results.items()
            if task_id in current_task_ids
        },
        conflicts,
    )
