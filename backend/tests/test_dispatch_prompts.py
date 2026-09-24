"""dispatch_prompts 纯函数测试：各阶段 prompt 组装与 XML 渲染的形状契约。"""

from __future__ import annotations

from types import SimpleNamespace

from app.schemas.dispatch import (
    DispatchPlanItem,
    TaskAcceptanceResult,
    TaskCommandEvidence,
    TaskFileEvidence,
    TaskResultReport,
    TaskTestEvidence,
)
from app.services.dispatch_prompts import (
    build_agent_hub_tool_guidance,
    build_aggregate_prompt,
    build_orchestrator_aggregate_prompt,
    build_orchestrator_plan_prompt,
    build_sub_agent_prompt,
    render_task_result_report_xml,
    SubAgentPromptData,
)
from app.services.dispatch_file_writes import FileWriteConflict, FileWriteContributor
from app.schemas.dispatch import DispatchTaskInput
from app.services.dispatch_scheduler import ResolvedTaskInput


def _agent_stub(**overrides) -> SimpleNamespace:
    base = dict(
        id="ag_worker_1",
        name="小工",
        capabilities=["docs"],
        tool_names=["write_artifact"],
        description="整理文档",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _task(**overrides) -> DispatchPlanItem:
    base = dict(id="t1", agentId="ag_worker_1", task="梳理需求要点")
    base.update(overrides)
    return DispatchPlanItem(**base)


# ─── plan 阶段 system prompt ────────────────────────────────


class TestOrchestratorPlanPrompt:
    def test_lists_available_agents_with_json_fields(self):
        prompt = build_orchestrator_plan_prompt(
            "你是编排者", [_agent_stub()], "sandbox"
        )
        assert 'id: "ag_worker_1"' in prompt
        assert 'name: "小工"' in prompt
        assert 'capabilities: ["docs"]' in prompt
        assert 'tools: ["write_artifact"]' in prompt
        assert 'description: "整理文档"' in prompt
        assert "## 你的工作流" in prompt
        assert "## 依赖关系" in prompt

    def test_local_workspace_rules_are_spliced_twice(self):
        prompt = build_orchestrator_plan_prompt("你是编排者", [], "local")
        marker = "## 本地 workspace 规划规则"
        assert prompt.count(marker) == 2
        assert "不要用 write_artifact 代替源码落盘" in prompt

    def test_sandbox_mode_has_no_local_rules(self):
        prompt = build_orchestrator_plan_prompt("你是编排者", [], "sandbox")
        assert "## 本地 workspace 规划规则" not in prompt

    def test_empty_agent_list_shows_placeholder(self):
        prompt = build_orchestrator_plan_prompt("你是编排者", [], "sandbox")
        assert "## 可用 Agent\n（无）" in prompt


class TestOrchestratorAggregatePrompt:
    def test_declares_aggregate_stage(self):
        prompt = build_orchestrator_aggregate_prompt("你是编排者")
        assert prompt.startswith("你是编排者")
        assert "聚合阶段" in prompt
        assert "不要再调用 plan_tasks" in prompt
        assert '<artifact_ref id="art_xxx"/>' in prompt


# ─── 工具调用规范 ───────────────────────────────────────────


class TestToolGuidance:
    def test_plan_stage_tools_have_plan_section(self):
        guidance = build_agent_hub_tool_guidance(
            "mock", ["plan_tasks", "ask_user"], "sandbox"
        )
        assert "### plan_tasks" in guidance
        assert "### ask_user" in guidance
        # 计划阶段没有文件工具段
        assert "### workspace 文件与命令工具" not in guidance

    def test_regular_tools_have_report_section(self):
        guidance = build_agent_hub_tool_guidance(
            "mock", ["fs_write", "bash", "report_task_result"], "sandbox"
        )
        assert "### report_task_result" in guidance
        assert "### workspace 文件与命令工具" in guidance
        assert "### plan_tasks" not in guidance

    def test_local_mode_section_for_workspace_tools(self):
        guidance = build_agent_hub_tool_guidance("mock", ["fs_write", "bash"], "local")
        assert "## 本地项目模式" in guidance

    def test_no_tools_gives_empty_guidance(self):
        assert build_agent_hub_tool_guidance("mock", [], "sandbox") == ""


# ─── 子 Agent prompt ────────────────────────────────────────


class TestSubAgentPrompt:
    def test_assembles_context_and_task_sections(self):
        prompt = build_sub_agent_prompt(
            SubAgentPromptData(
                task=_task(
                    acceptanceCriteria=["要点覆盖完整"],
                    targetPaths=["docs/notes.md"],
                ),
                resolved_inputs=[
                    ResolvedTaskInput(
                        input=DispatchTaskInput(fromTaskId="t0", outputId="prd"),
                        type="document",
                        artifact_id="art_1",
                        missing=False,
                    )
                ],
                workspace_mode="sandbox",
                summary_block="<conversation_summary covered_until_message_id=\"msg_1\">\n之前聊过的事\n</conversation_summary>",
            )
        )
        assert prompt.startswith("<context>")
        assert "<your_task>\n梳理需求要点\n</your_task>" in prompt
        assert "covered_until_message_id" in prompt
        assert "<required_inputs>" in prompt
        assert "<expected_outputs>" not in prompt  # 没声明就不出段落
        assert "<acceptance_criteria>" in prompt
        assert "<evidence_contract>" in prompt  # targetPaths 触发证据契约
        # 没有上游/已有产物时保留「（无）」占位
        assert "（无）" in prompt
        assert "report_task_result" in prompt

    def test_local_mode_line_present_only_for_local(self):
        local = build_sub_agent_prompt(
            SubAgentPromptData(task=_task(), resolved_inputs=[], workspace_mode="local")
        )
        sandbox = build_sub_agent_prompt(
            SubAgentPromptData(task=_task(), resolved_inputs=[], workspace_mode="sandbox")
        )
        assert "directly modify the current local workspace" in local
        assert "directly modify the current local workspace" not in sandbox


# ─── 聚合 prompt 与任务报告 XML ──────────────────────────────


class TestAggregatePrompt:
    def test_wraps_request_results_and_conflicts(self):
        plan = [
            _task(id="t1", agentId="ag_worker_1", task="梳理需求要点"),
            _task(id="t2", agentId="ag_worker_2", task="整理要点清单", dependsOn=["t1"]),
        ]
        from app.services.dispatch_scheduler import DispatchTaskResult

        task_results = {
            "t1": DispatchTaskResult(
                status="complete",
                run_id="run_1",
                task_report=TaskResultReport(status="complete", summary="搞定"),
            ),
            "t2": DispatchTaskResult(status="failed", error="门禁未过"),
        }
        conflicts = [
            FileWriteConflict(
                path="/workspace/docs/shared.md",
                contributors=[
                    FileWriteContributor(
                        taskId="t1", agentId="ag_worker_1", runId="run_1"
                    ),
                    FileWriteContributor(
                        taskId="t2", agentId="ag_worker_2", runId="run_2"
                    ),
                ],
            )
        ]
        prompt = build_aggregate_prompt(
            "做一份要点文档", plan, task_results, conflicts, "/workspace", {}
        )
        assert "<user_request>做一份要点文档</user_request>" in prompt
        assert '<result task="t1" agent="ag_worker_1" status="complete">' in prompt
        assert '<result task="t2" agent="ag_worker_2" status="failed" error="门禁未过">' in prompt
        assert "<file_conflicts>" in prompt
        assert "多个并行子任务写了同一文件" in prompt
        # 冲突路径转成 workspace 相对路径
        assert 'path="docs/shared.md"' in prompt
        assert "请基于以上结果给用户输出最终总结消息。" in prompt

    def test_omits_file_conflicts_section_when_clean(self):
        prompt = build_aggregate_prompt("需求", [], {}, [], "/workspace", {})
        assert "<file_conflicts>" not in prompt


class TestTaskReportXml:
    def test_renders_full_report_with_null_exit_code(self):
        report = TaskResultReport(
            status="complete",
            summary="已<整理>完毕",
            acceptanceResults=[
                TaskAcceptanceResult(
                    criterion="要点齐全", passed=True, evidence="自检通过"
                )
            ],
            filesChanged=[TaskFileEvidence(path="docs/a.md", action="created")],
            commandsRun=[
                TaskCommandEvidence(command="pnpm lint", exitCode=None, timedOut=False)
            ],
            tests=[TaskTestEvidence(command="pnpm test", passed=True)],
            blockers=None,
        )
        xml = render_task_result_report_xml(report)
        assert '<task_report status="complete">' in xml
        assert "已&lt;整理&gt;完毕" in xml  # 正文转义
        assert 'passed="true"' in xml
        # JS String(null) 语义：空退出码渲染成 "null" 字面量
        assert 'exitCode="null"' in xml
        assert 'timedOut="false"' in xml
        assert "<test " in xml
