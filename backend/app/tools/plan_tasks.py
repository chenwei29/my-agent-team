"""plan_tasks 工具：Orchestrator 专用的「输出端工具」。

本身无副作用，只校验拆解计划的结构并回 ack。真正的调度执行由
AgentRunner 在看到 plan_tasks 工具调用时接管（后续阶段实现）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.tools.types import ToolContext, ToolDef, ToolResult

TaskKind = Literal["code", "test", "review", "design", "doc", "analysis"]
OutputType = Literal["web_app", "document", "image", "ppt", "project"]


class _ExpectedOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    type: OutputType
    required: bool | None = None
    description: str | None = None


class _TaskInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    from_task_id: str = Field(alias="fromTaskId", min_length=1)
    output_id: str = Field(alias="outputId", min_length=1)
    required: bool | None = None
    description: str | None = None


class _RequiredCommand(BaseModel):
    model_config = ConfigDict(extra="ignore")

    command: str = Field(min_length=1)
    description: str | None = None
    cwd: str | None = None
    timeout_ms: int | None = Field(default=None, alias="timeoutMs", gt=0)


class _PlanTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    agent_id: str = Field(alias="agentId", min_length=1)
    task: str = Field(min_length=1)
    task_kind: TaskKind | None = Field(default=None, alias="taskKind")
    depends_on: list[str] | None = Field(default=None, alias="dependsOn")
    expected_outputs: list[_ExpectedOutput] | None = Field(default=None, alias="expectedOutputs")
    inputs: list[_TaskInput] | None = None
    acceptance_criteria: list[str] | None = Field(
        default=None, alias="acceptanceCriteria", min_length=1
    )
    target_paths: list[str] | None = Field(default=None, alias="targetPaths", min_length=1)
    expected_workspace_changes: list[str] | None = Field(
        default=None, alias="expectedWorkspaceChanges", min_length=1
    )
    required_commands: list[_RequiredCommand] | None = Field(
        default=None, alias="requiredCommands"
    )
    required_evidence: list[str] | None = Field(
        default=None, alias="requiredEvidence", min_length=1
    )


class PlanTasksArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reasoning: str = Field(min_length=1)
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
                        "id": {"type": "string", "description": "Sub-task id, use t1/t2/t3 format"},
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
                        },
                        "dependsOn": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ids of prerequisite tasks.",
                        },
                        "expectedOutputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["id", "type"],
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": ["web_app", "document", "image", "ppt", "project"],
                                    },
                                    "required": {"type": "boolean"},
                                    "description": {"type": "string"},
                                },
                            },
                        },
                        "inputs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["fromTaskId", "outputId"],
                                "properties": {
                                    "fromTaskId": {"type": "string"},
                                    "outputId": {"type": "string"},
                                    "required": {"type": "boolean"},
                                    "description": {"type": "string"},
                                },
                            },
                        },
                        "acceptanceCriteria": {"type": "array", "items": {"type": "string"}},
                        "targetPaths": {"type": "array", "items": {"type": "string"}},
                        "expectedWorkspaceChanges": {"type": "array", "items": {"type": "string"}},
                        "requiredCommands": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["command"],
                                "properties": {
                                    "command": {"type": "string"},
                                    "description": {"type": "string"},
                                    "cwd": {"type": "string"},
                                    "timeoutMs": {"type": "number"},
                                },
                            },
                        },
                        "requiredEvidence": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    },
    handler=_handle,
)
