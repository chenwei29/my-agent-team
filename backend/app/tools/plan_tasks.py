"""plan_tasks 工具：Orchestrator 专用的「输出端工具」。

本身无副作用，只校验拆解计划的结构并回 ack。真正的调度执行由
AgentRunner 在看到 plan_tasks 工具调用时接管（解析 + 语义校验走
services/dispatch_plan.py，那里会给出带任务定位的错误文案）。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.tools.types import ToolContext, ToolDef, ToolResult

TaskKind = Literal["code", "test", "review", "design", "doc", "analysis"]
OutputType = Literal["web_app", "document", "image", "ppt", "project"]

# 数组项「非空串」：与工具 schema 的描述口径一致（空串直接拒）
NonEmptyStr = Annotated[str, Field(min_length=1)]


class _ExpectedOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: NonEmptyStr
    type: OutputType
    required: bool | None = None
    description: str | None = None


class _TaskInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    from_task_id: NonEmptyStr = Field(alias="fromTaskId")
    output_id: NonEmptyStr = Field(alias="outputId")
    required: bool | None = None
    description: str | None = None


class _RequiredCommand(BaseModel):
    model_config = ConfigDict(extra="ignore")

    command: NonEmptyStr
    description: str | None = None
    cwd: NonEmptyStr | None = None
    timeout_ms: int | None = Field(default=None, alias="timeoutMs", gt=0)


class _PlanTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: NonEmptyStr
    agent_id: NonEmptyStr = Field(alias="agentId")
    task: NonEmptyStr
    task_kind: TaskKind | None = Field(default=None, alias="taskKind")
    depends_on: list[str] | None = Field(default=None, alias="dependsOn")
    expected_outputs: list[_ExpectedOutput] | None = Field(default=None, alias="expectedOutputs")
    inputs: list[_TaskInput] | None = None
    acceptance_criteria: list[NonEmptyStr] | None = Field(default=None, alias="acceptanceCriteria")
    target_paths: list[NonEmptyStr] | None = Field(default=None, alias="targetPaths")
    expected_workspace_changes: list[NonEmptyStr] | None = Field(
        default=None, alias="expectedWorkspaceChanges"
    )
    required_commands: list[_RequiredCommand] | None = Field(
        default=None, alias="requiredCommands"
    )
    required_evidence: list[NonEmptyStr] | None = Field(default=None, alias="requiredEvidence")


class PlanTasksArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reasoning: NonEmptyStr
    tasks: list[_PlanTask] = Field(min_length=1)


async def _handle(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = PlanTasksArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid plan: {err}")

    # 实际执行由 AgentRunner 接管；这里仅做格式校验并回 ack。
    return ToolResult(
        ok=True,
        value={"acknowledged": True, "taskCount": len(parsed.tasks)},
    )


PLAN_TASKS_TOOL = ToolDef(
    name="plan_tasks",
    description=(
        "Decompose the user request into sub-tasks and dispatch them to other agents "
        "in this group. Output a complete plan in a single call; do NOT call this tool "
        "multiple times."
    ),
    parameters={
        "type": "object",
        "required": ["reasoning", "tasks"],
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Brief explanation of why this decomposition makes sense, 3 sentences max"
                ),
            },
            "tasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "agentId", "task"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Sub-task id, use t1/t2/t3 format",
                        },
                        "agentId": {
                            "type": "string",
                            "description": (
                                "Agent id that should execute this task. Must come from the "
                                "available list."
                            ),
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "Concrete, self-contained instruction for that agent. The agent "
                                "will not see the full group history."
                            ),
                        },
                        "taskKind": {
                            "type": "string",
                            "enum": ["code", "test", "review", "design", "doc", "analysis"],
                            "description": (
                                "Kind of work. Use code/test/review/design/doc/analysis to help "
                                "AgentRunner apply evidence expectations."
                            ),
                        },
                        "dependsOn": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Ids of prerequisite tasks. Omit when the task can start "
                                "immediately."
                            ),
                        },
                        "expectedOutputs": {
                            "type": "array",
                            "description": (
                                "Artifacts this task must create for downstream handoff or user "
                                "inspection. Code implementation tasks should declare a required "
                                "project output. Omit for text-only work such as review, "
                                "validation, diagnosis, status check, explanation, or summary."
                            ),
                            "items": {
                                "type": "object",
                                "required": ["id", "type"],
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": (
                                            "Symbolic output key within this task, not an "
                                            "artifact id."
                                        ),
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": ["web_app", "document", "image", "ppt", "project"],
                                        "description": (
                                            "Expected artifact type. Use project for workspace "
                                            "code trees; project is created by AgentHub from "
                                            "fs_write evidence, not by write_artifact."
                                        ),
                                    },
                                    "required": {
                                        "type": "boolean",
                                        "description": (
                                            "Whether this handoff output is expected by the "
                                            "plan. Defaults to true. Required project outputs "
                                            "on code tasks are hard completion gates."
                                        ),
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": (
                                            "Short description of what this output should "
                                            "contain."
                                        ),
                                    },
                                },
                            },
                        },
                        "inputs": {
                            "type": "array",
                            "description": (
                                "Upstream artifacts this task must consume. AgentRunner "
                                "validates these against upstream expectedOutputs and compiles "
                                "them into dependencies."
                            ),
                            "items": {
                                "type": "object",
                                "required": ["fromTaskId", "outputId"],
                                "properties": {
                                    "fromTaskId": {
                                        "type": "string",
                                        "description": (
                                            "Upstream task id that produces the artifact."
                                        ),
                                    },
                                    "outputId": {
                                        "type": "string",
                                        "description": (
                                            "The upstream expectedOutputs.id to consume."
                                        ),
                                    },
                                    "required": {
                                        "type": "boolean",
                                        "description": (
                                            "Whether this input is required. Defaults to true."
                                        ),
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Why this input is needed.",
                                    },
                                },
                            },
                        },
                        "acceptanceCriteria": {
                            "type": "array",
                            "description": (
                                "Concrete completion checks for this task. Use this for "
                                "text-only/review/validation tasks instead of expectedOutputs. "
                                "The child agent must report each item through "
                                "report_task_result.acceptanceResults."
                            ),
                            "items": {"type": "string"},
                        },
                        "targetPaths": {
                            "type": "array",
                            "description": (
                                "Workspace file or directory paths this task is expected to "
                                "inspect, create, or change. Use relative paths when possible."
                            ),
                            "items": {"type": "string"},
                        },
                        "expectedWorkspaceChanges": {
                            "type": "array",
                            "description": (
                                "Plain-language list of expected workspace changes. Required "
                                "for non-trivial code tasks."
                            ),
                            "items": {"type": "string"},
                        },
                        "requiredCommands": {
                            "type": "array",
                            "description": (
                                "Commands that AgentHub must run successfully before accepting "
                                "this task as complete, such as pnpm test or mvn compile. Use "
                                "cwd instead of cd for subdirectories."
                            ),
                            "items": {
                                "type": "object",
                                "required": ["command"],
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "description": (
                                            "Exact command expected to run. Keep it focused on "
                                            "the verification step; use cwd for subdirectories."
                                        ),
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": (
                                            "Short reason this command verifies the task."
                                        ),
                                    },
                                    "cwd": {
                                        "type": "string",
                                        "description": (
                                            'Optional workspace-relative directory to run the '
                                            'command in, such as "frontend" or "backend".'
                                        ),
                                    },
                                    "timeoutMs": {
                                        "type": "number",
                                        "description": (
                                            "Optional timeout in milliseconds. Use a larger "
                                            "value for dependency install or compilation."
                                        ),
                                    },
                                },
                            },
                        },
                        "requiredEvidence": {
                            "type": "array",
                            "description": (
                                "Evidence statements the child must provide in "
                                "report_task_result before the task can complete."
                            ),
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
    handler=_handle,
)
