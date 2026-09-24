"""调度计划（dispatch plan）与任务结果上报（task result report）的数据结构。

对齐前端 `web/src/shared/types.ts` 的 DispatchPlanItem / TaskResultReport 段：
字段名**手写 camelCase**（事件与工具出入参都是直传 JSON，不走 alias 转换）。

两个消费面：
- StreamEvent 的 `dispatch.*` 事件 payload（schemas/events.py 直接引用这里的模型）；
- `plan_tasks` / `report_task_result` 工具与调度纯函数（services/dispatch_plan.py 等）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

DispatchTaskKind = Literal["code", "test", "review", "design", "doc", "analysis"]
DispatchExpectedOutputType = Literal["web_app", "document", "image", "ppt", "diagram", "project"]
TaskResultReportStatus = Literal["complete", "failed", "blocked"]
DispatchTaskStatus = Literal[
    "pending", "running", "complete", "failed", "aborted", "skipped"
]
# 终态：pending / running 不算
DispatchTaskEndStatus = Literal["complete", "failed", "aborted", "skipped"]

_TASK_CONTRACT_EXTRA = ConfigDict(extra="forbid")


class DispatchExpectedOutput(BaseModel):
    """子任务对外交付的一项产物契约。id 是计划内的符号 key，不是数据库 artifact id。"""

    model_config = _TASK_CONTRACT_EXTRA
    id: str
    type: DispatchExpectedOutputType
    required: bool | None = None
    description: str | None = None


class DispatchTaskInput(BaseModel):
    """子任务要消费的上游产物引用（fromTaskId.outputId）。"""

    model_config = _TASK_CONTRACT_EXTRA
    fromTaskId: str
    outputId: str
    required: bool | None = None
    description: str | None = None


class DispatchRequiredCommand(BaseModel):
    """任务完成前必须跑通的命令（证据门禁的命令项）。"""

    model_config = _TASK_CONTRACT_EXTRA
    command: str
    description: str | None = None
    cwd: str | None = None
    timeoutMs: int | None = None


class DispatchPlanItem(BaseModel):
    """分派计划里的一条子任务：执行者、任务描述、依赖与验收契约。"""

    model_config = _TASK_CONTRACT_EXTRA
    id: str
    agentId: str
    task: str
    taskKind: DispatchTaskKind | None = None
    dependsOn: list[str] | None = None
    expectedOutputs: list[DispatchExpectedOutput] | None = None
    inputs: list[DispatchTaskInput] | None = None
    acceptanceCriteria: list[str] | None = None
    targetPaths: list[str] | None = None
    expectedWorkspaceChanges: list[str] | None = None
    requiredCommands: list[DispatchRequiredCommand] | None = None
    requiredEvidence: list[str] | None = None


class PendingDispatchPlan(BaseModel):
    """等待用户审批的整份分派计划（dispatch.plan.pending 的 payload）。"""

    model_config = _TASK_CONTRACT_EXTRA
    id: str
    conversationId: str
    agentId: str
    runId: str
    plan: list[DispatchPlanItem]
    createdAt: int


class TaskAcceptanceResult(BaseModel):
    """一条验收标准的判定结果：criterion 必须与计划里的文本一致。"""

    model_config = _TASK_CONTRACT_EXTRA
    criterion: str
    passed: bool
    evidence: str


class TaskFileEvidence(BaseModel):
    model_config = _TASK_CONTRACT_EXTRA
    path: str
    action: Literal["created", "modified", "deleted", "verified"] | None = None


class TaskCommandEvidence(BaseModel):
    model_config = _TASK_CONTRACT_EXTRA
    command: str
    exitCode: int | None
    cwd: str | None = None
    timedOut: bool | None = None
    summary: str | None = None


class TaskTestEvidence(BaseModel):
    model_config = _TASK_CONTRACT_EXTRA
    command: str
    passed: bool
    summary: str | None = None


class TaskResultReport(BaseModel):
    """子 Agent 收尾时经 report_task_result 上报的结构化结果。"""

    model_config = _TASK_CONTRACT_EXTRA
    status: TaskResultReportStatus
    summary: str
    acceptanceResults: list[TaskAcceptanceResult] | None = None
    filesChanged: list[TaskFileEvidence] | None = None
    commandsRun: list[TaskCommandEvidence] | None = None
    tests: list[TaskTestEvidence] | None = None
    blockers: list[str] | None = None


# ─── 运行期工具证据（供证据门禁判定）────────────────────────
class RunFileEvidence(BaseModel):
    """一次 run 内经 fs_write 落盘的文件记录。"""

    model_config = ConfigDict(extra="ignore")
    path: str
    absolutePath: str
    bytes: int | None = None
    applied: Literal["auto", "review"] | None = None


class RunCommandEvidence(BaseModel):
    """一次 run 内执行过的命令（bash 工具或 requiredCommands 门禁补跑）。"""

    model_config = ConfigDict(extra="ignore")
    command: str
    cwd: str
    exitCode: int | None
    timedOut: bool
    isError: bool
    prepare: bool | None = None
    error: str | None = None


class RunToolEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    fileWrites: list[RunFileEvidence] = []
    commands: list[RunCommandEvidence] = []


class CompileInferredDependency(BaseModel):
    """compile_dispatch_plan 从任务文本推断出的依赖（供日志 / 测试观察）。"""

    model_config = ConfigDict(extra="ignore")
    taskId: str
    dependsOn: list[str]
    reason: str


class ReplanTaskView(BaseModel):
    """重规划视图里的一条任务状态（should_replan / build_replan_context 的输入）。"""

    model_config = ConfigDict(extra="ignore")
    taskId: str
    agentId: str
    status: Literal["complete", "failed", "skipped", "aborted"]
    error: str | None = None


class ReplanConflictView(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    taskIds: list[str]


def dump_plan_item(item: DispatchPlanItem) -> dict[str, Any]:
    """计划项 → 事件/响应用的 JSON dict（省略未设置的可选字段）。"""
    return item.model_dump(exclude_none=True)
