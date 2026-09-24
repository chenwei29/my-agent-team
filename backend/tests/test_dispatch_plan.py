"""派发计划纯函数测试：plan_tasks 解析、语义校验、依赖推断/编译、环检测、重规划上下文。

用例名与断言值是行为契约：错误文案、推断出的依赖顺序、编译补全的产物契约
被 Orchestrator 运行期与前端计划卡片共同消费，改动前先想清楚两边的影响。
"""

from __future__ import annotations

import json
import re

import pytest

from app.schemas.dispatch import (
    DispatchPlanItem,
    ReplanConflictView,
    ReplanTaskView,
    dump_plan_item,
)
from app.services.dispatch_plan import (
    CODE_TASK_PROJECT_OUTPUT_DESCRIPTION,
    CODE_TASK_RUNNABLE_ACCEPTANCE_CRITERION,
    CODE_TASK_RUNNABLE_REQUIRED_EVIDENCE,
    assert_acyclic_dispatch_plan,
    build_replan_context,
    collect_dependency_closure,
    compile_dispatch_plan,
    extract_plan_tasks_tool_args,
    parse_dispatch_plan_tool_args,
    should_replan,
    task_expects_artifact,
    validate_dispatch_plan,
)

agents = [
    {"id": "ag_pm"},
    {"id": "ag_designer"},
    {"id": "ag_frontend"},
    {"id": "ag_reviewer"},
]

plan_args = {
    "reasoning": "Split into implementation and review.",
    "tasks": [{"id": "t1", "agentId": "ag_frontend", "task": "Build UI"}],
}


def task(
    id: str,
    agentId: str,
    dependsOn: list[str] | None = None,
    instruction: str | None = None,
) -> DispatchPlanItem:
    return DispatchPlanItem(
        id=id,
        agentId=agentId,
        task=instruction if instruction is not None else f"Do {id}",
        dependsOn=dependsOn,
    )


def raises(fragment: str):
    """断言抛 ValueError 且消息包含 fragment（与 toThrow 的子串语义一致）。"""
    return pytest.raises(ValueError, match=re.escape(fragment))


class TestParseDispatchPlanToolArgs:
    def test_parses_valid_plan_tasks_args(self):
        assert [dump_plan_item(t) for t in parse_dispatch_plan_tool_args({
            "tasks": [
                {"id": "t1", "agentId": "ag_pm", "task": "Write PRD"},
                {"id": "t2", "agentId": "ag_frontend", "task": "Build UI", "dependsOn": ["t1"]},
            ],
        })] == [
            {"id": "t1", "agentId": "ag_pm", "task": "Write PRD"},
            {"id": "t2", "agentId": "ag_frontend", "task": "Build UI", "dependsOn": ["t1"]},
        ]

    def test_parses_task_contracts_for_artifact_handoff(self):
        assert [dump_plan_item(t) for t in parse_dispatch_plan_tool_args({
            "tasks": [
                {
                    "id": "t1",
                    "agentId": "ag_pm",
                    "task": "Write PRD",
                    "expectedOutputs": [
                        {
                            "id": "prd",
                            "type": "document",
                            "description": "Product requirements",
                        },
                    ],
                    "acceptanceCriteria": ["Includes P0 scope"],
                },
                {
                    "id": "t2",
                    "agentId": "ag_frontend",
                    "task": "Build UI",
                    "inputs": [{"fromTaskId": "t1", "outputId": "prd"}],
                    "expectedOutputs": [{"id": "web_app", "type": "web_app"}],
                },
            ],
        })] == [
            {
                "id": "t1",
                "agentId": "ag_pm",
                "task": "Write PRD",
                "expectedOutputs": [
                    {
                        "id": "prd",
                        "type": "document",
                        "description": "Product requirements",
                    },
                ],
                "acceptanceCriteria": ["Includes P0 scope"],
            },
            {
                "id": "t2",
                "agentId": "ag_frontend",
                "task": "Build UI",
                "inputs": [{"fromTaskId": "t1", "outputId": "prd"}],
                "expectedOutputs": [{"id": "web_app", "type": "web_app"}],
            },
        ]

    def test_parses_evidence_contract_fields(self):
        assert [dump_plan_item(t) for t in parse_dispatch_plan_tool_args({
            "tasks": [
                {
                    "id": "t1",
                    "agentId": "ag_frontend",
                    "task": "Implement endpoint",
                    "taskKind": "code",
                    "targetPaths": ["src/foo.ts"],
                    "expectedWorkspaceChanges": ["Add foo handler"],
                    "requiredCommands": [
                        {
                            "command": "pnpm test src/foo.test.ts",
                            "cwd": "frontend",
                            "timeoutMs": 300000,
                        },
                    ],
                    "requiredEvidence": ["测试命令 exitCode=0"],
                },
            ],
        })] == [
            {
                "id": "t1",
                "agentId": "ag_frontend",
                "task": "Implement endpoint",
                "taskKind": "code",
                "targetPaths": ["src/foo.ts"],
                "expectedWorkspaceChanges": ["Add foo handler"],
                "requiredCommands": [
                    {
                        "command": "pnpm test src/foo.test.ts",
                        "cwd": "frontend",
                        "timeoutMs": 300000,
                    },
                ],
                "requiredEvidence": ["测试命令 exitCode=0"],
            },
        ]

    def test_parses_project_expected_outputs_for_workspace_code_trees(self):
        assert [dump_plan_item(t) for t in parse_dispatch_plan_tool_args({
            "tasks": [
                {
                    "id": "t1",
                    "agentId": "ag_frontend",
                    "task": "Implement frontend project",
                    "taskKind": "code",
                    "expectedOutputs": [{"id": "project", "type": "project", "required": True}],
                },
            ],
        })] == [
            {
                "id": "t1",
                "agentId": "ag_frontend",
                "task": "Implement frontend project",
                "taskKind": "code",
                "expectedOutputs": [{"id": "project", "type": "project", "required": True}],
            },
        ]

    def test_rejects_malformed_tool_args(self):
        with raises("tasks array"):
            parse_dispatch_plan_tool_args(None)
        with raises("task at index 0 must be an object"):
            parse_dispatch_plan_tool_args({"tasks": ["bad"]})
        with raises("task at index 0 id must be a non-empty string"):
            parse_dispatch_plan_tool_args({"tasks": [{"id": "", "agentId": "ag_pm", "task": "x"}]})
        with raises("dependsOn must be an array"):
            parse_dispatch_plan_tool_args(
                {"tasks": [{"id": "t1", "agentId": "ag_pm", "task": "x", "dependsOn": "t0"}]}
            )
        with raises("dependsOn[0] must be a non-empty string"):
            parse_dispatch_plan_tool_args(
                {"tasks": [{"id": "t1", "agentId": "ag_pm", "task": "x", "dependsOn": [1]}]}
            )
        with raises('task "t1" taskKind must be one of'):
            parse_dispatch_plan_tool_args(
                {"tasks": [{"id": "t1", "agentId": "ag_pm", "task": "x", "taskKind": "unknown"}]}
            )
        with raises("requiredCommands[0].timeoutMs must be a positive integer"):
            parse_dispatch_plan_tool_args({
                "tasks": [
                    {
                        "id": "t1",
                        "agentId": "ag_pm",
                        "task": "x",
                        "requiredCommands": [{"command": "pnpm build", "timeoutMs": -1}],
                    },
                ],
            })


class TestValidateDispatchPlan:
    def test_accepts_a_valid_acyclic_plan(self):
        plan = [
            task("t1", "ag_pm"),
            task("t2", "ag_frontend", ["t1"]),
            task("t3", "ag_reviewer", ["t2"]),
        ]

        validate_dispatch_plan(plan, agents, "ag_orchestrator")

    def test_rejects_empty_plans_and_duplicate_task_ids(self):
        with raises("tasks must not be empty"):
            validate_dispatch_plan([], agents, "ag_orchestrator")
        with raises("duplicate task id(s): t1"):
            validate_dispatch_plan(
                [task("t1", "ag_pm"), task("t1", "ag_frontend")], agents, "ag_orchestrator"
            )

    def test_rejects_unavailable_or_recursive_agent_targets(self):
        with raises("dispatches to the orchestrator itself"):
            validate_dispatch_plan([task("t1", "ag_orchestrator")], agents, "ag_orchestrator")
        with raises('references unavailable agentId "ag_missing"'):
            validate_dispatch_plan([task("t1", "ag_missing")], agents, "ag_orchestrator")

    def test_rejects_invalid_dependencies(self):
        with raises("cannot depend on itself"):
            validate_dispatch_plan([task("t1", "ag_pm", ["t1"])], agents, "ag_orchestrator")
        with raises('depends on unknown task "t0"'):
            validate_dispatch_plan([task("t1", "ag_pm", ["t0"])], agents, "ag_orchestrator")
        with raises('lists duplicate dependency "t1"'):
            validate_dispatch_plan(
                [task("t1", "ag_pm"), task("t2", "ag_frontend", ["t1", "t1"])],
                agents,
                "ag_orchestrator",
            )

    def test_accepts_dependencies_on_resolved_external_tasks_during_replan(self):
        validate_dispatch_plan(
            [task("t4", "ag_frontend", ["t2"])],
            agents,
            "ag_orchestrator",
            [task("t2", "ag_pm")],
        )

    def test_rejects_invalid_task_contracts(self):
        with raises('duplicate expected output "prd"'):
            validate_dispatch_plan(
                [
                    DispatchPlanItem(
                        id="t1",
                        agentId="ag_pm",
                        task="Do t1",
                        expectedOutputs=[
                            {"id": "prd", "type": "document"},
                            {"id": "prd", "type": "document"},
                        ],
                    ),
                ],
                agents,
                "ag_orchestrator",
            )

        with raises('input references unknown output "missing" from task "t1"'):
            validate_dispatch_plan(
                [
                    DispatchPlanItem(
                        id="t1",
                        agentId="ag_pm",
                        task="Do t1",
                        expectedOutputs=[{"id": "prd", "type": "document"}],
                    ),
                    DispatchPlanItem(
                        id="t2",
                        agentId="ag_frontend",
                        task="Do t2",
                        inputs=[{"fromTaskId": "t1", "outputId": "missing"}],
                    ),
                ],
                agents,
                "ag_orchestrator",
            )

    def test_accepts_inputs_from_resolved_external_tasks_during_replan(self):
        validate_dispatch_plan(
            [
                DispatchPlanItem(
                    id="t4",
                    agentId="ag_frontend",
                    task="Do t4",
                    inputs=[{"fromTaskId": "t2", "outputId": "prd"}],
                ),
            ],
            agents,
            "ag_orchestrator",
            [
                DispatchPlanItem(
                    id="t2",
                    agentId="ag_pm",
                    task="Do t2",
                    expectedOutputs=[{"id": "prd", "type": "document"}],
                ),
            ],
        )

    def test_rejects_circular_dependencies(self):
        plan = [task("t1", "ag_pm", ["t2"]), task("t2", "ag_frontend", ["t1"])]

        with raises("circular dependency t1 -> t2 -> t1"):
            validate_dispatch_plan(plan, agents, "ag_orchestrator")


class TestExtractPlanTasksToolArgs:
    def test_recognizes_native_agenthub_plan_tasks_tool_calls(self):
        assert extract_plan_tasks_tool_args({"toolName": "plan_tasks", "args": plan_args}) is plan_args

    def test_recognizes_claude_code_mcp_plan_tasks_tool_calls(self):
        assert (
            extract_plan_tasks_tool_args(
                {"toolName": "mcp__agenthub__plan_tasks", "args": plan_args}
            )
            is plan_args
        )

    def test_unwraps_codex_mcp_plan_tasks_tool_call_arguments(self):
        assert extract_plan_tasks_tool_args({
            "toolName": "codex_mcp_agenthub_plan_tasks",
            "args": {"server": "agenthub", "tool": "plan_tasks", "arguments": plan_args},
        }) is plan_args

    def test_unwraps_json_encoded_codex_mcp_plan_tasks_tool_call_arguments(self):
        assert extract_plan_tasks_tool_args({
            "toolName": "codex_mcp_agenthub_plan_tasks",
            "args": {
                "server": "agenthub",
                "tool": "plan_tasks",
                "arguments": json.dumps(plan_args),
            },
        }) == plan_args

    def test_ignores_unrelated_tool_calls(self):
        assert (
            extract_plan_tasks_tool_args({"toolName": "write_artifact", "args": plan_args}) is None
        )


class TestCompileDispatchPlan:
    def test_adds_required_project_output_and_runnable_evidence_to_code_tasks(self):
        plan, _ = compile_dispatch_plan([
            DispatchPlanItem(
                id="t1",
                agentId="ag_frontend",
                task="Implement the frontend app",
                taskKind="code",
            ),
        ])

        assert {
            "id": "project",
            "type": "project",
            "required": True,
            "description": CODE_TASK_PROJECT_OUTPUT_DESCRIPTION,
        } in [o.model_dump(exclude_none=True) for o in (plan[0].expectedOutputs or [])]
        assert CODE_TASK_RUNNABLE_ACCEPTANCE_CRITERION in (plan[0].acceptanceCriteria or [])
        assert CODE_TASK_RUNNABLE_REQUIRED_EVIDENCE in (plan[0].requiredEvidence or [])

    def test_forces_declared_project_outputs_on_code_tasks_to_be_required(self):
        plan, _ = compile_dispatch_plan([
            DispatchPlanItem(
                id="t1",
                agentId="ag_frontend",
                task="Implement the backend project",
                taskKind="code",
                expectedOutputs=[{"id": "workspace", "type": "project", "required": False}],
            ),
        ])

        assert [o.model_dump(exclude_none=True) for o in (plan[0].expectedOutputs or [])] == [
            {
                "id": "workspace",
                "type": "project",
                "required": True,
                "description": CODE_TASK_PROJECT_OUTPUT_DESCRIPTION,
            },
        ]

    def test_does_not_add_project_outputs_to_explicit_review_tasks(self):
        plan, _ = compile_dispatch_plan([
            DispatchPlanItem(
                id="t1",
                agentId="ag_reviewer",
                task="Review the implementation and summarize risks",
                taskKind="review",
            ),
        ])

        assert plan[0].expectedOutputs is None
        assert plan[0].acceptanceCriteria is None
        assert plan[0].requiredEvidence is None

    def test_infers_missing_dependencies_from_task_id_artifact_references(self):
        plan, inferred_dependencies = compile_dispatch_plan([
            task("t1", "ag_pm", None, "请产出 PRD 文档，并写入 artifact。"),
            task("t2", "ag_frontend", None, "读取 t1 产物后实现 web_app artifact。"),
        ])

        assert plan[1].dependsOn == ["t1"]
        assert [d.model_dump(exclude_none=True) for d in inferred_dependencies] == [
            {
                "taskId": "t2",
                "dependsOn": ["t1"],
                "reason": "task text references earlier task output",
            },
        ]
        validate_dispatch_plan(plan, agents, "ag_orchestrator")

    def test_infers_the_prd_to_ui_to_frontend_to_reviewer_incident_pattern(self):
        plan, _ = compile_dispatch_plan([
            task("t1", "ag_pm", None, "输出一份 PRD 文档，并写入 artifact。"),
            task("t2", "ag_designer", None, "读取 PRD artifact 后输出一份 UI 设计方案。"),
            task("t3", "ag_frontend", None, "读取 PRD 和 UI 设计，输出一个 web_app artifact。"),
            task(
                "t4",
                "ag_reviewer",
                None,
                "审查前端工程师产出的 web_app artifact，检查是否符合 PRD 和 UI 设计，并输出审查报告。",
            ),
        ])

        assert [{"id": item.id, "dependsOn": item.dependsOn} for item in plan] == [
            {"id": "t1", "dependsOn": None},
            {"id": "t2", "dependsOn": ["t1"]},
            {"id": "t3", "dependsOn": ["t1", "t2"]},
            {"id": "t4", "dependsOn": ["t1", "t2", "t3"]},
        ]

    def test_preserves_explicit_dependencies_while_adding_missing_review_predecessors(self):
        plan, _ = compile_dispatch_plan([
            task("t1", "ag_pm", None, "输出一份 PRD 文档，并写入 artifact。"),
            task("t2", "ag_designer", ["t1"], "读取 PRD artifact 后输出一份 UI 设计方案。"),
            task("t3", "ag_frontend", ["t2"], "读取 UI 设计，输出一个 web_app artifact。"),
            task("t4", "ag_reviewer", ["t3"], "审查实现是否符合 PRD 和 UI 设计，并输出审查报告。"),
        ])

        assert plan[3].dependsOn == ["t3", "t1", "t2"]

    def test_compiles_task_inputs_into_dependencies(self):
        plan, _ = compile_dispatch_plan([
            DispatchPlanItem(
                id="t1",
                agentId="ag_pm",
                task="Do t1",
                expectedOutputs=[{"id": "prd", "type": "document"}],
            ),
            DispatchPlanItem(
                id="t2",
                agentId="ag_frontend",
                task="Do t2",
                inputs=[{"fromTaskId": "t1", "outputId": "prd"}],
            ),
        ])

        assert plan[1].dependsOn == ["t1"]
        validate_dispatch_plan(plan, agents, "ag_orchestrator")


class TestCollectDependencyClosure:
    def test_returns_transitive_dependencies_in_upstream_order(self):
        plan = [
            task("t1", "ag_pm"),
            task("t2", "ag_designer", ["t1"]),
            task("t3", "ag_frontend", ["t2"]),
            task("t4", "ag_reviewer", ["t3"]),
        ]

        assert collect_dependency_closure(plan, "t4") == ["t1", "t2", "t3"]


class TestTaskExpectsArtifact:
    def test_distinguishes_artifact_producing_tasks_from_read_only_tasks(self):
        assert task_expects_artifact(task("t1", "ag_pm", None, "输出一个 document/markdown artifact。")) is True
        assert task_expects_artifact(task("t2", "ag_frontend", None, "请实现一个完整的响应式网页应用。")) is True
        assert task_expects_artifact(task("t2", "ag_reviewer", None, "读取 artifact 并总结主要问题。")) is False


class TestAssertAcyclicDispatchPlan:
    def test_accepts_linear_dags(self):
        assert_acyclic_dispatch_plan([task("t1", "ag_pm"), task("t2", "ag_frontend", ["t1"])])

    def test_detects_self_and_multi_node_cycles(self):
        with raises("circular dependency t1 -> t1"):
            assert_acyclic_dispatch_plan([task("t1", "ag_pm", ["t1"])])
        with raises("circular dependency t1 -> t2 -> t1"):
            assert_acyclic_dispatch_plan(
                [task("t1", "ag_pm", ["t2"]), task("t2", "ag_frontend", ["t1"])]
            )


class TestShouldReplan:
    def test_false_when_all_tasks_complete_and_no_conflicts(self):
        assert should_replan(
            [
                ReplanTaskView(taskId="t1", agentId="ag_pm", status="complete"),
                ReplanTaskView(taskId="t2", agentId="ag_frontend", status="complete"),
            ],
            [],
        ) is False

    def test_true_when_a_task_failed_or_was_skipped(self):
        assert should_replan(
            [
                ReplanTaskView(taskId="t1", agentId="ag_pm", status="complete"),
                ReplanTaskView(
                    taskId="t2", agentId="ag_frontend", status="failed", error="no artifact"
                ),
            ],
            [],
        ) is True

    def test_true_when_there_is_a_write_conflict_even_if_all_complete(self):
        assert should_replan(
            [ReplanTaskView(taskId="t1", agentId="ag_a", status="complete")],
            [ReplanConflictView(path="index.html", taskIds=["t1", "t2"])],
        ) is True


class TestBuildReplanContext:
    def test_lists_complete_plus_failed_tasks_plus_conflicts_and_instructs_remediation(self):
        ctx = build_replan_context(
            [
                ReplanTaskView(taskId="t1", agentId="ag_pm", status="complete"),
                ReplanTaskView(
                    taskId="t2", agentId="ag_frontend", status="failed", error="missing artifact"
                ),
            ],
            [ReplanConflictView(path="index.html", taskIds=["t2", "t3"])],
        )
        assert "<previous_round_results>" in ctx
        assert 'id="t1" agent="ag_pm" status="complete"' in ctx
        assert 'status="failed"' in ctx
        assert "missing artifact" in ctx
        assert "<file_conflicts>" in ctx
        assert "index.html" in ctx
        assert "plan_tasks" in ctx
