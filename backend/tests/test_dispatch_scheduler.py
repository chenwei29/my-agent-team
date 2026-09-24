"""DAG 调度器行为测试：固定计划直接喂调度器，绕开 LLM。

覆盖的调度契约：波内并发、依赖顺序、失败传播 skipped、中止标 aborted、
同波次写冲突只上报、每个任务恰好一条 dispatch.end、子任务证据门禁与重试、
requiredCommands 补跑与汇总证据。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from app.schemas.dispatch import (
    DispatchPlanItem,
    DispatchRequiredCommand,
    RunCommandEvidence,
    RunToolEvidence,
    TaskResultReport,
)
from app.services import dispatch_scheduler
from app.services.dispatch_file_writes import record_file_write
from app.services.dispatch_scheduler import (
    DagContext,
    DispatchTaskResult,
    Semaphore,
    VerificationCommandResult,
    build_task_continuation_context,
    execute_dag,
    expand_required_command,
    run_child_task,
)
from app.utils.abort import AbortSignal


def task(task_id: str, **overrides) -> DispatchPlanItem:
    data = {"id": task_id, "agentId": "ag_worker", "task": "Write the report"}
    data.update(overrides)
    return DispatchPlanItem(**data)


def ctx_for(signal: AbortSignal | None = None, **overrides) -> DagContext:
    data = {
        "parent_run_id": "run_parent",
        "conversation_id": "conv_test",
        "trigger_message_id": "msg_trigger",
        "signal": signal or AbortSignal(),
    }
    data.update(overrides)
    return DagContext(**data)


@pytest.fixture
def events(monkeypatch) -> list:
    captured: list = []
    monkeypatch.setattr(dispatch_scheduler.event_bus, "publish", captured.append)
    return captured


def complete_result(run_id: str | None = "run_x", **overrides) -> DispatchTaskResult:
    return DispatchTaskResult(
        run_id=run_id, status="complete", artifact_ids=["art_1"], **overrides
    )


class FakeRuntime:
    """按脚本吐尝试结果的子任务运行时；记录启动次数与补跑命令。"""

    def __init__(self, outcomes: list, evidence: RunToolEvidence | None = None) -> None:
        self.outcomes = list(outcomes)
        self.evidence = evidence or RunToolEvidence()
        self.launches = 0
        self.commands: list[tuple[str, bool]] = []
        self.recorded: list[RunCommandEvidence] = []
        self.project_artifact_id: str | None = None
        self.prepare: object = ...
        self.command_ok = True

    async def build_sub_agent_prompt(self, task_, resolved_inputs, upstream, plan) -> str:
        return f"prompt:{task_.id}"

    def launch_attempt(self, task_, prompt, ctx):
        index = self.launches
        self.launches += 1
        run_id = f"run_{task_.id}_{index}"
        outcome = self.outcomes[index] if index < len(self.outcomes) else self.outcomes[-1]
        if callable(outcome):
            result = outcome(task_, run_id)
        else:
            result = replace(
                outcome,
                run_id=run_id,
                artifact_ids=list(outcome.artifact_ids),
                output_artifacts=dict(outcome.output_artifacts),
            )

        async def _wait() -> DispatchTaskResult:
            await asyncio.sleep(0)
            return result

        return run_id, _wait()

    def get_evidence(self, run_id: str) -> RunToolEvidence:
        return self.evidence

    def record_command_evidence(self, run_id: str, evidence: RunCommandEvidence) -> None:
        # 生产实现写入按 run 的证据表；这里同样入证据面，门禁才看得到补跑汇总
        self.recorded.append(evidence)
        self.evidence.commands.append(evidence)

    async def workspace_ready(self) -> bool:
        return True

    def effective_cwd(self) -> str:
        return "/ws"

    def build_prepare_command(self, required_cwd):
        if self.prepare is ...:
            return None
        if isinstance(self.prepare, Exception):
            raise self.prepare
        return self.prepare

    async def execute_command(self, spec, *, timeout_ms, prepare, task, run_id, ctx):
        self.commands.append((spec.command, prepare))
        return VerificationCommandResult(
            command=spec.command,
            ok=self.command_ok,
            exit_code=0 if self.command_ok else 1,
            timed_out=False,
            cwd=spec.cwd,
            prepare=prepare,
        )

    async def create_project_artifact(self, task_, evidence, result) -> str | None:
        return self.project_artifact_id


# ─── execute_dag：波次 / 失败传播 / 中止 / 冲突 ─────────────
class TestExecuteDag:
    async def test_three_task_two_wave_plan_runs_in_dependency_order_concurrently(self):
        """t1 先跑；t2/t3 同波并发（两个都在 t1 完成后才起跑）。"""
        plan = [
            task("t1"),
            task("t2", dependsOn=["t1"]),
            task("t3", dependsOn=["t1"]),
        ]
        order: list[str] = []
        in_flight = 0
        max_in_flight = 0

        async def run_child(task_, upstream, plan_context, ctx):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            order.append(f"start:{task_.id}")
            await asyncio.sleep(0.01)
            order.append(f"end:{task_.id}")
            in_flight -= 1
            return complete_result(run_id=f"run_{task_.id}")

        results, conflicts = await execute_dag(plan, ctx_for(), run_child)

        assert list(results) == ["t1", "t2", "t3"]
        assert all(r.status == "complete" for r in results.values())
        assert order.index("end:t1") < order.index("start:t2")
        assert order.index("end:t1") < order.index("start:t3")
        assert max_in_flight == 2  # 波内并发
        assert conflicts == []

    async def test_failed_task_skips_dependents_but_independents_run(self, events):
        plan = [
            task("t1"),
            task("t2", dependsOn=["t1"]),
            task("t3"),
        ]

        async def run_child(task_, upstream, plan_context, ctx):
            if task_.id == "t1":
                return DispatchTaskResult(
                    run_id="run_t1", status="failed", error="boom"
                )
            return complete_result(run_id=f"run_{task_.id}")

        results, _ = await execute_dag(plan, ctx_for(), run_child)

        assert results["t2"].status == "skipped"
        assert results["t2"].error == (
            'Skipped because upstream task(s) did not complete for task "t2": t1:failed'
        )
        assert results["t3"].status == "complete"
        ends = {e.taskId: e for e in events if e.type == "dispatch.end"}
        assert ends["t2"].status == "skipped"
        assert ends["t2"].childRunId is None

    async def test_abort_marks_remaining_tasks_and_raises(self, events):
        signal = AbortSignal()
        plan = [task("t1"), task("t2", dependsOn=["t1"])]

        async def run_child(task_, upstream, plan_context, ctx):
            signal.abort()  # 第一波执行期间中止父 run
            return complete_result(run_id="run_t1")

        with pytest.raises(RuntimeError, match="Orchestrator run aborted"):
            await execute_dag(plan, ctx_for(signal=signal), run_child)

        ends = {e.taskId: e for e in events if e.type == "dispatch.end"}
        assert ends["t2"].status == "aborted"
        assert ends["t2"].error == 'Aborted before task "t2" started'

    async def test_same_wave_writes_with_different_content_reported_as_conflict(self):
        plan = [task("t1"), task("t2")]

        async def run_child(task_, upstream, plan_context, ctx):
            run_id = f"run_{task_.id}"
            record_file_write(run_id, "/ws/index.html", f"content-{task_.id}")
            return complete_result(run_id=run_id)

        results, conflicts = await execute_dag(plan, ctx_for(), run_child)

        assert len(conflicts) == 1
        assert conflicts[0].path == "/ws/index.html"
        assert sorted(c.taskId for c in conflicts[0].contributors) == ["t1", "t2"]

    async def test_identical_concurrent_writes_are_not_a_conflict(self):
        plan = [task("t1"), task("t2")]

        async def run_child(task_, upstream, plan_context, ctx):
            run_id = f"run_{task_.id}"
            record_file_write(run_id, "/ws/index.html", "same")
            return complete_result(run_id=run_id)

        _, conflicts = await execute_dag(plan, ctx_for(), run_child)
        assert conflicts == []

    async def test_dispatch_end_emitted_exactly_once_per_task(self, events):
        plan = [task("t1"), task("t2", dependsOn=["t1"]), task("t3", dependsOn=["t2"])]

        async def run_child(task_, upstream, plan_context, ctx):
            return DispatchTaskResult(run_id=f"run_{task_.id}", status="failed")

        await execute_dag(plan, ctx_for(), run_child)

        ends = [e for e in events if e.type == "dispatch.end"]
        assert sorted(e.taskId for e in ends) == ["t1", "t2", "t3"]
        assert [e.status for e in ends if e.taskId == "t2"] == ["skipped"]

    async def test_seed_results_supply_completed_external_dependencies(self):
        """补救轮：依赖已在上一轮 complete 的外部任务 → 本任务照常执行。"""
        plan = [task("t2", dependsOn=["t0"])]
        seed = {"t0": complete_result(run_id="run_t0")}

        async def run_child(task_, upstream, plan_context, ctx):
            assert upstream["t0"].status == "complete"
            return complete_result(run_id="run_t2")

        results, _ = await execute_dag(plan, ctx_for(seed_results=seed), run_child)
        assert results["t2"].status == "complete"

    async def test_circular_dependencies_raise(self):
        plan = [task("t1", dependsOn=["t2"]), task("t2", dependsOn=["t1"])]

        async def run_child(task_, upstream, plan_context, ctx):  # pragma: no cover
            raise AssertionError("should not run")

        with pytest.raises(RuntimeError, match="Circular dependency or unresolved task in plan"):
            await execute_dag(plan, ctx_for(), run_child)


# ─── run_child_task：门禁 / 重试 / 事件 / requiredCommands ──
class TestRunChildTask:
    async def test_missing_required_input_skips_without_launching(self, events):
        t = task(
            "t2",
            dependsOn=["t1"],
            inputs=[{"fromTaskId": "t1", "outputId": "prd", "required": True}],
        )
        runtime = FakeRuntime([complete_result()])
        upstream = {"t1": complete_result(run_id="run_t1")}  # 没绑 prd 产物

        result = await run_child_task(t, upstream, [task("t1"), t], ctx_for(), runtime)

        assert result.status == "skipped"
        assert result.error == (
            'Skipped because required input artifact(s) were missing for task "t2": t1.prd'
        )
        assert runtime.launches == 0
        ends = [e for e in events if e.type == "dispatch.end"]
        assert ends == []  # dispatch.end 由 execute_dag 发；单独调用不发

    async def test_report_without_evidence_gate_retries_then_fails(self, events):
        """自称 complete 但没上报 → 门禁翻 failed，重试耗尽后带次数说明。"""
        t = task("t1")
        runtime = FakeRuntime([complete_result()])
        result = await run_child_task(t, {}, [t], ctx_for(), runtime)

        assert runtime.launches == dispatch_scheduler.MAX_CHILD_TASK_ATTEMPTS
        assert result.status == "failed"
        assert result.error == (
            'Task "t1" did not satisfy completion gates after '
            f"{dispatch_scheduler.MAX_CHILD_TASK_ATTEMPTS} attempt(s). "
            'Last error: Task "t1" completed without report_task_result'
        )
        assert result.run_ids == [f"run_t1_{i}" for i in range(runtime.launches)]
        starts = [e for e in events if e.type == "dispatch.start"]
        assert len(starts) == runtime.launches
        assert all(e.taskId == "t1" for e in starts)

    async def test_complete_report_passes_and_binds_single_output(self):
        t = task(
            "t1",
            taskKind="doc",
            expectedOutputs=[{"id": "doc", "type": "document"}],
        )
        report = TaskResultReport(status="complete", summary="done")
        runtime = FakeRuntime([complete_result(task_report=report)])

        result = await run_child_task(t, {}, [t], ctx_for(), runtime)

        assert result.status == "complete"
        assert result.output_artifacts == {"doc": "art_1"}
        assert result.run_ids == ["run_t1_0"]

    async def test_blocked_report_stops_retrying(self):
        t = task("t1", taskKind="review")
        report = TaskResultReport(status="blocked", summary="stuck", blockers=["no access"])
        runtime = FakeRuntime([complete_result(task_report=report)])

        result = await run_child_task(t, {}, [t], ctx_for(), runtime)

        assert runtime.launches == 1
        assert result.task_report is not None
        assert result.task_report.status == "blocked"

    async def test_project_artifact_binds_and_gates_required_project_output(self):
        t = task(
            "t1",
            taskKind="code",
            expectedOutputs=[{"id": "app", "type": "project"}],
        )
        report = TaskResultReport(status="complete", summary="built")
        # code 任务门禁要求成功的验证命令证据
        evidence = RunToolEvidence(
            commands=[
                RunCommandEvidence(
                    command="pnpm test", cwd="/ws", exitCode=0, timedOut=False, isError=False
                )
            ]
        )
        runtime = FakeRuntime(
            [replace(complete_result(task_report=report), artifact_ids=[])], evidence=evidence
        )
        runtime.project_artifact_id = "art_project"

        result = await run_child_task(t, {}, [t], ctx_for(), runtime)

        assert result.status == "complete"
        assert result.output_artifacts == {"app": "art_project"}
        assert "art_project" in result.artifact_ids

    async def test_project_output_without_verification_fails_code_gate(self):
        t = task("t1", expectedOutputs=[{"id": "app", "type": "project"}])
        report = TaskResultReport(status="complete", summary="built")
        runtime = FakeRuntime([complete_result(task_report=report)], evidence=RunToolEvidence())

        result = await run_child_task(t, {}, [t], ctx_for(), runtime)

        assert result.status == "failed"
        assert "runnable verification command evidence" in result.error

    async def test_required_commands_record_summary_evidence(self):
        t = task(
            "t1",
            taskKind="review",
            requiredCommands=[
                DispatchRequiredCommand(
                    command="cd web && pnpm lint && pnpm test", timeoutMs=1000
                )
            ],
        )
        report = TaskResultReport(status="complete", summary="ok")
        runtime = FakeRuntime([complete_result(task_report=report)])
        from app.services.dispatch_scheduler import PreparedCommand

        runtime.prepare = PreparedCommand(command="pnpm install", cwd="web")

        result = await run_child_task(t, {}, [t], ctx_for(), runtime)

        assert result.status == "complete"
        # 命令里没有 install 才补 prepare；cd 前缀拆出 cwd + 两条顺序命令
        assert runtime.commands == [
            ("pnpm install", True),
            ("pnpm lint", False),
            ("pnpm test", False),
        ]
        assert len(runtime.recorded) == 1
        summary = runtime.recorded[0]
        assert summary.command == "cd web && pnpm lint && pnpm test"
        assert summary.exitCode == 0
        assert summary.isError is False

    async def test_required_commands_with_install_skip_prepare(self):
        t = task(
            "t1",
            taskKind="review",
            requiredCommands=[DispatchRequiredCommand(command="npm install && npm test")],
        )
        report = TaskResultReport(status="complete", summary="ok")
        runtime = FakeRuntime([complete_result(task_report=report)])
        from app.services.dispatch_scheduler import PreparedCommand

        runtime.prepare = PreparedCommand(command="pnpm install", cwd=None)

        await run_child_task(t, {}, [t], ctx_for(), runtime)

        assert runtime.commands == [("npm install", False), ("npm test", False)]

    async def test_required_commands_stop_at_first_failure_and_record_error(self):
        t = task(
            "t1",
            taskKind="review",
            requiredCommands=[DispatchRequiredCommand(command="pnpm lint && pnpm test")],
        )
        report = TaskResultReport(status="complete", summary="ok")
        runtime = FakeRuntime([complete_result(task_report=report)])
        runtime.command_ok = False

        result = await run_child_task(t, {}, [t], ctx_for(), runtime)

        # 每轮重试都补跑；第一条失败后同一轮内第二条不跑
        assert runtime.commands == [("pnpm lint", False)] * dispatch_scheduler.MAX_CHILD_TASK_ATTEMPTS
        assert ("pnpm test", False) not in runtime.commands
        summary = runtime.recorded[0]
        assert summary.exitCode == 1
        assert summary.isError is True
        # 失败的命令证据进面后被门禁拦下（每轮重试都带着这条失败证据）
        assert result.status == "failed"
        assert "has failed command evidence" in result.error

    async def test_prepare_failure_skips_required_entry_without_summary(self):
        t = task(
            "t1",
            taskKind="review",
            requiredCommands=[DispatchRequiredCommand(command="pnpm lint")],
        )
        report = TaskResultReport(status="complete", summary="ok")
        runtime = FakeRuntime([complete_result(task_report=report)])
        runtime.prepare = ValueError("cwd is outside the workspace")

        result = await run_child_task(t, {}, [t], ctx_for(), runtime)

        assert runtime.commands == []
        assert runtime.recorded == []
        # 没有成功命令证据 → 门禁拦下
        assert result.status == "failed"
        assert "missing successful command evidence" in result.error

    async def test_continuation_prompt_carries_previous_attempt_context(self):
        from app.services.dispatch_scheduler import ChildAttemptEvaluation

        t = task("t1", targetPaths=["src/app.ts"])
        evaluation = ChildAttemptEvaluation(
            raw_result=complete_result(),
            result=replace(complete_result(), status="failed", error="gate failed"),
            evidence=RunToolEvidence(),
            verification_results=[
                VerificationCommandResult(
                    command="pnpm test", ok=False, exit_code=1, timed_out=False, output="boom"
                )
            ],
        )
        text = build_task_continuation_context(t, evaluation, 1, 4)
        assert "<previous_attempt>" in text
        assert "<status>failed</status>" in text
        assert "<missing_report>true</missing_report>" in text
        assert "<path>src/app.ts</path>" in text
        assert 'ok="false"' in text
        assert "<output>boom</output>" in text


# ─── 并发闸门 / 命令展开 ───────────────────────────────────
class TestSemaphoreAndExpand:
    async def test_semaphore_rejects_acquire_on_abort_while_queued(self):
        semaphore = Semaphore(1)
        signal = AbortSignal()
        release = await semaphore.acquire(AbortSignal())

        waiting = asyncio.ensure_future(semaphore.acquire(signal))
        await asyncio.sleep(0)  # 让 waiter 入队
        signal.abort()

        with pytest.raises(RuntimeError, match="Semaphore acquire aborted"):
            await waiting
        release()

    async def test_expand_required_command_splits_cd_and_chains(self):
        required = DispatchRequiredCommand(command='cd "web app" && pnpm lint && pnpm test')
        expanded = expand_required_command(required)
        assert expanded.cwd == "web app"
        assert expanded.commands == ["pnpm lint", "pnpm test"]

    async def test_expand_required_command_keeps_explicit_cwd(self):
        required = DispatchRequiredCommand(command="cd web && pnpm lint", cwd="api")
        expanded = expand_required_command(required)
        assert expanded.cwd == "api"
        assert expanded.commands == ["cd web", "pnpm lint"]
