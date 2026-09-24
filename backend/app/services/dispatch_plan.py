"""派发计划的解析 + 编译 + 语义校验 + 环检测（纯函数模块）。

只依赖 schemas/dispatch 的数据结构，不牵 DB / 网络，便于单测。
执行调度（DAG、子 run）留在 agent_runner；本模块负责：

- `parse_dispatch_plan_tool_args`：把 LLM 调 plan_tasks 的原始 args 解析成计划项，
  错误文案逐条带任务定位（`task "t2" dependsOn[0] …`），方便前端直接展示；
- `compile_dispatch_plan`：补推断依赖（文本里引用了前序任务产物）、把 inputs 编译成
  dependsOn、给代码实现任务补 project 产物与可验证验收契约；
- `validate_dispatch_plan`：语义校验（重复 id / 未知 agent / 自依赖 / 成环 / 契约引用），
  依赖是唯一执行顺序契约，成环必须在这里拦住，否则拓扑排序死循环。

校验错误一律 `Invalid dispatch plan: …` 前缀，运行期把消息原样写进 run.end.error。
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.schemas.dispatch import (
    CompileInferredDependency,
    DispatchExpectedOutput,
    DispatchExpectedOutputType,
    DispatchPlanItem,
    DispatchRequiredCommand,
    DispatchTaskInput,
    DispatchTaskKind,
    ReplanConflictView,
    ReplanTaskView,
)

# 有序元组：错误文案 `must be one of …` 用这个顺序拼（不是字母序）
WRITABLE_ARTIFACT_TYPES: tuple[str, ...] = ("web_app", "document", "image", "ppt", "diagram")
EXPECTED_OUTPUT_TYPES: tuple[str, ...] = (*WRITABLE_ARTIFACT_TYPES, "project")
DISPATCH_TASK_KINDS: tuple[str, ...] = ("code", "test", "review", "design", "doc", "analysis")

CODE_TASK_PROJECT_OUTPUT_ID = "project"
CODE_TASK_PROJECT_OUTPUT_DESCRIPTION = "Workspace project files written by this code task"
CODE_TASK_RUNNABLE_ACCEPTANCE_CRITERION = "项目构建/编译验证通过（至少一条非准备验证命令 exitCode=0）"
CODE_TASK_RUNNABLE_REQUIRED_EVIDENCE = "至少一条构建/编译/测试/类型检查命令 exitCode=0"
PLAN_TASKS_TOOL_NAME = "plan_tasks"

ArtifactTopic = str  # 'prd' | 'ui_design' | 'frontend'


# ─── plan_tasks 工具调用识别 / 解析 ─────────────────────────


def extract_plan_tasks_tool_args(event: dict[str, Any]) -> Any | None:
    """从 tool.call 事件里取出 plan_tasks 的 args；不是 plan_tasks 返回 None。

    兼容三种工具名形态：原生 `plan_tasks`、Claude MCP 前缀 `mcp__agenthub__*`、
    Codex MCP 前缀 `codex_mcp_agenthub_*`（Codex 会把参数包一层 {tool, arguments}，
    arguments 还可能是 JSON 字符串）。
    """
    tool_name = event.get("toolName")
    args = event.get("args")
    if tool_name == PLAN_TASKS_TOOL_NAME:
        return args
    if tool_name == f"mcp__agenthub__{PLAN_TASKS_TOOL_NAME}":
        return args
    if tool_name == f"codex_mcp_agenthub_{PLAN_TASKS_TOOL_NAME}":
        return _read_codex_mcp_tool_arguments(args)
    if isinstance(tool_name, str) and (
        tool_name.endswith(f"__{PLAN_TASKS_TOOL_NAME}")
        or tool_name.endswith(f"_{PLAN_TASKS_TOOL_NAME}")
    ):
        return args
    return None


def _read_codex_mcp_tool_arguments(args: Any) -> Any:
    if not _is_record(args) or args.get("tool") != PLAN_TASKS_TOOL_NAME:
        return args
    raw_arguments = args.get("arguments")
    if not isinstance(raw_arguments, str):
        return raw_arguments
    try:
        return json.loads(raw_arguments)
    except (json.JSONDecodeError, ValueError):
        return raw_arguments


def parse_dispatch_plan_tool_args(args: Any) -> list[DispatchPlanItem]:
    """plan_tasks 的原始 args → 计划项列表；形状不对直接抛 `Invalid dispatch plan: …`。"""
    if not _is_record(args) or not isinstance(args.get("tasks"), list):
        raise ValueError("Invalid dispatch plan: plan_tasks args must include a tasks array")

    return [_parse_plan_task(raw, index) for index, raw in enumerate(args["tasks"])]


def _parse_plan_task(raw: Any, index: int) -> DispatchPlanItem:
    if not _is_record(raw):
        raise ValueError(f"Invalid dispatch plan: task at index {index} must be an object")
    task_id = _read_non_empty_string(raw.get("id"), f"task at index {index} id")
    agent_id = _read_non_empty_string(raw.get("agentId"), f'task "{task_id}" agentId')
    task = _read_non_empty_string(raw.get("task"), f'task "{task_id}" instruction')
    task_kind = _read_optional_task_kind(raw.get("taskKind", _MISSING), f'task "{task_id}" taskKind')

    depends_on: list[str] | None = None
    if "dependsOn" in raw:
        if not isinstance(raw["dependsOn"], list):
            raise ValueError(f'Invalid dispatch plan: task "{task_id}" dependsOn must be an array')
        depends_on = [
            _read_non_empty_string(dep, f'task "{task_id}" dependsOn[{dep_index}]')
            for dep_index, dep in enumerate(raw["dependsOn"])
        ]

    expected_outputs: list[DispatchExpectedOutput] | None = None
    if "expectedOutputs" in raw:
        if not isinstance(raw["expectedOutputs"], list):
            raise ValueError(
                f'Invalid dispatch plan: task "{task_id}" expectedOutputs must be an array'
            )
        expected_outputs = [
            _parse_expected_output(task_id, output, output_index)
            for output_index, output in enumerate(raw["expectedOutputs"])
        ]

    inputs: list[DispatchTaskInput] | None = None
    if "inputs" in raw:
        if not isinstance(raw["inputs"], list):
            raise ValueError(f'Invalid dispatch plan: task "{task_id}" inputs must be an array')
        inputs = [
            _parse_task_input(task_id, item, item_index)
            for item_index, item in enumerate(raw["inputs"])
        ]

    acceptance_criteria = _read_optional_string_list(
        raw.get("acceptanceCriteria", _MISSING), task_id, "acceptanceCriteria"
    )
    target_paths = _read_optional_string_list(raw.get("targetPaths", _MISSING), task_id, "targetPaths")
    expected_workspace_changes = _read_optional_string_list(
        raw.get("expectedWorkspaceChanges", _MISSING), task_id, "expectedWorkspaceChanges"
    )
    required_evidence = _read_optional_string_list(
        raw.get("requiredEvidence", _MISSING), task_id, "requiredEvidence"
    )

    required_commands: list[DispatchRequiredCommand] | None = None
    if "requiredCommands" in raw:
        if not isinstance(raw["requiredCommands"], list):
            raise ValueError(
                f'Invalid dispatch plan: task "{task_id}" requiredCommands must be an array'
            )
        required_commands = [
            _parse_required_command(task_id, command, command_index)
            for command_index, command in enumerate(raw["requiredCommands"])
        ]

    # 空数组 / 空串可选字段一律省略（与事件 payload 的形状约定一致）
    return DispatchPlanItem(
        id=task_id,
        agentId=agent_id,
        task=task,
        taskKind=task_kind,
        dependsOn=depends_on or None,
        expectedOutputs=expected_outputs or None,
        inputs=inputs or None,
        acceptanceCriteria=acceptance_criteria or None,
        targetPaths=target_paths or None,
        expectedWorkspaceChanges=expected_workspace_changes or None,
        requiredCommands=required_commands or None,
        requiredEvidence=required_evidence or None,
    )


def _parse_expected_output(task_id: str, output: Any, output_index: int) -> DispatchExpectedOutput:
    label = f'task "{task_id}" expectedOutputs[{output_index}]'
    if not _is_record(output):
        raise ValueError(f"Invalid dispatch plan: {label} must be an object")
    return DispatchExpectedOutput(
        id=_read_non_empty_string(output.get("id"), f"{label}.id"),
        type=_read_expected_output_type(output.get("type"), f"{label}.type"),
        required=_read_optional_boolean(output.get("required", _MISSING), f"{label}.required"),
        description=_read_optional_string(output.get("description", _MISSING), f"{label}.description"),
    )


def _parse_task_input(task_id: str, item: Any, item_index: int) -> DispatchTaskInput:
    label = f'task "{task_id}" inputs[{item_index}]'
    if not _is_record(item):
        raise ValueError(f"Invalid dispatch plan: {label} must be an object")
    return DispatchTaskInput(
        fromTaskId=_read_non_empty_string(item.get("fromTaskId"), f"{label}.fromTaskId"),
        outputId=_read_non_empty_string(item.get("outputId"), f"{label}.outputId"),
        required=_read_optional_boolean(item.get("required", _MISSING), f"{label}.required"),
        description=_read_optional_string(item.get("description", _MISSING), f"{label}.description"),
    )


def _parse_required_command(
    task_id: str, command: Any, command_index: int
) -> DispatchRequiredCommand:
    label = f'task "{task_id}" requiredCommands[{command_index}]'
    if not _is_record(command):
        raise ValueError(f"Invalid dispatch plan: {label} must be an object")
    return DispatchRequiredCommand(
        command=_read_non_empty_string(command.get("command"), f"{label}.command"),
        description=_read_optional_string(command.get("description", _MISSING), f"{label}.description"),
        cwd=_read_optional_string(command.get("cwd", _MISSING), f"{label}.cwd"),
        timeoutMs=_read_optional_positive_integer(
            command.get("timeoutMs", _MISSING), f"{label}.timeoutMs"
        ),
    )


def _read_optional_string_list(value: Any, task_id: str, field: str) -> list[str] | None:
    if value is _MISSING:
        return None
    if not isinstance(value, list):
        raise ValueError(f'Invalid dispatch plan: task "{task_id}" {field} must be an array')
    return [
        _read_non_empty_string(item, f'task "{task_id}" {field}[{item_index}]')
        for item_index, item in enumerate(value)
    ]


def _read_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"Invalid dispatch plan: {label} must be a non-empty string")
    return value


# 可选字段的「没给」哨兵：与显式 null 区分 —— 显式 null 一律按类型错误报出去
_MISSING: Any = object()


def _read_optional_string(value: Any, label: str) -> str | None:
    if value is _MISSING:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid dispatch plan: {label} must be a string")
    trimmed = value.strip()
    return trimmed if trimmed else None


def _read_optional_boolean(value: Any, label: str) -> bool | None:
    if value is _MISSING:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"Invalid dispatch plan: {label} must be a boolean")
    return value


def _read_optional_positive_integer(value: Any, label: str) -> int | None:
    if value is _MISSING:
        return None
    # bool 是 int 子类，先排掉；整数值的 float（JSON 里 300000.0）也算合法
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"Invalid dispatch plan: {label} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"Invalid dispatch plan: {label} must be a positive integer")
    return value


def _read_optional_task_kind(value: Any, label: str) -> DispatchTaskKind | None:
    if value is _MISSING:
        return None
    if not isinstance(value, str) or value not in DISPATCH_TASK_KINDS:
        raise ValueError(
            f"Invalid dispatch plan: {label} must be one of {', '.join(DISPATCH_TASK_KINDS)}"
        )
    return value  # type: ignore[return-value]


def _read_expected_output_type(value: Any, label: str) -> DispatchExpectedOutputType:
    if not isinstance(value, str) or value not in EXPECTED_OUTPUT_TYPES:
        raise ValueError(
            f"Invalid dispatch plan: {label} must be one of {', '.join(EXPECTED_OUTPUT_TYPES)}"
        )
    return value  # type: ignore[return-value]


# ─── 语义校验 ───────────────────────────────────────────────


def validate_dispatch_plan(
    plan: list[DispatchPlanItem],
    available_agents: list[Any],
    orchestrator_agent_id: str,
    resolved_external_tasks: list[DispatchPlanItem] | None = None,
) -> None:
    """坏计划在这里给出清晰错误；通过不代表可执行（编译后还要再过一次）。"""
    if not plan:
        raise ValueError("Invalid dispatch plan: tasks must not be empty")

    available_agent_ids = {a.id if hasattr(a, "id") else a["id"] for a in available_agents}
    task_ids: set[str] = set()
    duplicate_task_ids: set[str] = set()
    for task in plan:
        if task.id in task_ids:
            duplicate_task_ids.add(task.id)
        task_ids.add(task.id)
    if duplicate_task_ids:
        raise ValueError(
            f"Invalid dispatch plan: duplicate task id(s): {', '.join(sorted(duplicate_task_ids))}"
        )

    task_by_id = {task.id: task for task in plan}
    external_by_id = {task.id: task for task in (resolved_external_tasks or [])}

    for task in plan:
        if task.agentId == orchestrator_agent_id:
            raise ValueError(
                f'Invalid dispatch plan: task "{task.id}" dispatches to the orchestrator itself, '
                "which would recurse"
            )
        if task.agentId not in available_agent_ids:
            raise ValueError(
                f'Invalid dispatch plan: task "{task.id}" references unavailable agentId "{task.agentId}"'
            )

        dep_ids: set[str] = set()
        for dep in task.dependsOn or []:
            if dep == task.id:
                raise ValueError(f'Invalid dispatch plan: task "{task.id}" cannot depend on itself')
            if dep in dep_ids:
                raise ValueError(
                    f'Invalid dispatch plan: task "{task.id}" lists duplicate dependency "{dep}"'
                )
            dep_ids.add(dep)
            if dep not in task_ids and dep not in external_by_id:
                raise ValueError(
                    f'Invalid dispatch plan: task "{task.id}" depends on unknown task "{dep}"'
                )

        output_ids: set[str] = set()
        for output in task.expectedOutputs or []:
            if output.id in output_ids:
                raise ValueError(
                    f'Invalid dispatch plan: task "{task.id}" lists duplicate expected output "{output.id}"'
                )
            output_ids.add(output.id)

        for task_input in task.inputs or []:
            if task_input.fromTaskId == task.id:
                raise ValueError(
                    f'Invalid dispatch plan: task "{task.id}" input cannot reference itself'
                )
            upstream = task_by_id.get(task_input.fromTaskId) or external_by_id.get(
                task_input.fromTaskId
            )
            if upstream is None:
                raise ValueError(
                    f'Invalid dispatch plan: task "{task.id}" input references unknown task '
                    f'"{task_input.fromTaskId}"'
                )
            if not any(o.id == task_input.outputId for o in upstream.expectedOutputs or []):
                raise ValueError(
                    f'Invalid dispatch plan: task "{task.id}" input references unknown output '
                    f'"{task_input.outputId}" from task "{task_input.fromTaskId}"'
                )

    assert_acyclic_dispatch_plan(plan)


def assert_acyclic_dispatch_plan(plan: list[DispatchPlanItem]) -> None:
    """DFS 着色找环；环路径按访问顺序拼进错误消息（`t1 -> t2 -> t1`）。"""
    by_id = {task.id: task for task in plan}
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            cycle_start = stack.index(task_id)
            cycle = [*stack[cycle_start:], task_id]
            raise ValueError(f"Invalid dispatch plan: circular dependency {' -> '.join(cycle)}")
        task = by_id.get(task_id)
        if task is None:
            return
        visiting.add(task_id)
        stack.append(task_id)
        for dep in task.dependsOn or []:
            visit(dep)
        stack.pop()
        visiting.remove(task_id)
        visited.add(task_id)

    for task in plan:
        visit(task.id)


# ─── 计划编译（推断依赖 + 代码任务契约归一化）───────────────────


def compile_dispatch_plan(
    plan: list[DispatchPlanItem],
) -> tuple[list[DispatchPlanItem], list[CompileInferredDependency]]:
    """返回 (编译后的计划, 推断出的依赖清单)。

    顺序契约只认 dependsOn；LLM 常把依赖写进 task 文本漏写字段，这里做确定性补全：
    仅从同一计划中**排在当前任务之前**的任务推断，保持顺序稳定、去重。
    """
    inferred_dependencies: list[CompileInferredDependency] = []
    compiled: list[DispatchPlanItem] = []

    for index, task in enumerate(plan):
        previous_tasks = plan[:index]
        inferred = _infer_dependencies_for_task(task, previous_tasks)
        explicit = list(task.dependsOn or [])
        input_deps = [item.fromTaskId for item in task.inputs or []]
        dependency_set = set(explicit)
        dependencies = list(explicit)
        additions = [dep for dep in inferred if dep not in dependency_set]
        for dep in additions:
            dependencies.append(dep)
            dependency_set.add(dep)
        for dep in input_deps:
            if dep in dependency_set:
                continue
            dependencies.append(dep)
            dependency_set.add(dep)

        merged = _clone_plan_item(task, dependsOn=dependencies or None)
        if additions:
            inferred_dependencies.append(
                CompileInferredDependency(
                    taskId=task.id,
                    dependsOn=additions,
                    reason="task text references earlier task output",
                )
            )
        compiled.append(normalize_task_contract(merged))

    return compiled, inferred_dependencies


def _clone_plan_item(
    task: DispatchPlanItem, *, dependsOn: list[str] | None
) -> DispatchPlanItem:
    data = task.model_dump(exclude_none=True)
    data["dependsOn"] = dependsOn
    return DispatchPlanItem(**data)


def normalize_task_contract(task: DispatchPlanItem) -> DispatchPlanItem:
    """代码实现任务补齐：required project 产物 + 可验证验收项 + 验证命令证据要求。"""
    if not is_code_implementation_task(task):
        return task
    data = task.model_dump(exclude_none=True)
    data["expectedOutputs"] = [
        o.model_dump(exclude_none=True)
        for o in _ensure_code_project_output(task.expectedOutputs or [])
    ]
    data["acceptanceCriteria"] = _append_unique(
        list(task.acceptanceCriteria or []), CODE_TASK_RUNNABLE_ACCEPTANCE_CRITERION
    )
    data["requiredEvidence"] = _append_unique(
        list(task.requiredEvidence or []), CODE_TASK_RUNNABLE_REQUIRED_EVIDENCE
    )
    return DispatchPlanItem(**data)


def is_code_implementation_task(task: DispatchPlanItem) -> bool:
    if task.taskKind is not None:
        return task.taskKind == "code"
    if any(o.type == "project" for o in task.expectedOutputs or []):
        return True
    if (task.targetPaths or task.expectedWorkspaceChanges) and not _is_review_task(task.task):
        return True
    return bool(_CODE_TASK_PATTERN.search(task.task)) and not _is_review_task(task.task)


_CODE_TASK_PATTERN = re.compile(
    r"(?:实现|开发|修复|改造|重构|搭建|脚手架|前端|后端|接口|组件|页面|代码|项目|工程|应用|构建|编译"
    r"|implement|develop|build|scaffold|frontend|backend|api|endpoint|component|page|code"
    r"|project|app|fix|refactor)",
    re.IGNORECASE,
)


def _ensure_code_project_output(
    outputs: list[DispatchExpectedOutput],
) -> list[DispatchExpectedOutput]:
    """已有 project 输出则强制 required=True；没有则追加一个默认 project 输出。"""
    has_project = False
    normalized: list[DispatchExpectedOutput] = []
    for output in outputs:
        if output.type != "project":
            normalized.append(output)
            continue
        has_project = True
        normalized.append(
            DispatchExpectedOutput(
                id=output.id,
                type="project",
                required=True,
                description=output.description or CODE_TASK_PROJECT_OUTPUT_DESCRIPTION,
            )
        )
    if has_project:
        return normalized
    normalized.append(
        DispatchExpectedOutput(
            id=_next_unique_output_id(outputs, CODE_TASK_PROJECT_OUTPUT_ID),
            type="project",
            required=True,
            description=CODE_TASK_PROJECT_OUTPUT_DESCRIPTION,
        )
    )
    return normalized


def _next_unique_output_id(outputs: list[DispatchExpectedOutput], preferred: str) -> str:
    used = {output.id for output in outputs}
    if preferred not in used:
        return preferred
    index = 2
    while True:
        candidate = f"{preferred}_{index}"
        if candidate not in used:
            return candidate
        index += 1


def _append_unique(values: list[str], value: str) -> list[str]:
    normalized = {item.strip() for item in values}
    return values if value in normalized else [*values, value]


def collect_dependency_closure(plan: list[DispatchPlanItem], task_id: str) -> list[str]:
    """任务的传递依赖闭包，按「上游在前」的拓扑序返回（不含任务自身）。"""
    by_id = {task.id: task for task in plan}
    task = by_id.get(task_id)
    if task is None:
        return []

    seen: set[str] = set()
    ordered: list[str] = []

    def visit(dep_id: str) -> None:
        if dep_id in seen:
            return
        dep = by_id.get(dep_id)
        if dep is None:
            return
        for nested in dep.dependsOn or []:
            visit(nested)
        seen.add(dep_id)
        ordered.append(dep_id)

    for dep in task.dependsOn or []:
        visit(dep)
    return ordered


def task_expects_artifact(task: DispatchPlanItem) -> bool:
    """任务是否预期产出 artifact（文本启发式，供依赖推断用）。"""
    if any(output.required is not False for output in task.expectedOutputs or []):
        return True
    text = task.task
    return (
        bool(_get_produced_artifact_topics(text))
        or bool(
            re.search(
                r"(?:输出|产出|写入|生成|创建|保存).{0,40}"
                r"(?:artifact|artifacts|产物|document|web_app|web app|diagram|mermaid|diff|code_file|markdown|文档|报告|网页|应用|代码|PRD|设计|图)",
                text,
                re.IGNORECASE,
            )
        )
        or bool(
            re.search(
                r"(?:artifact|artifacts|产物|document|web_app|web app|diagram|mermaid|diff|code_file|markdown|文档|报告|网页|应用|代码|PRD|设计|图).{0,40}"
                r"(?:输出|产出|写入|生成|创建|保存)",
                text,
                re.IGNORECASE,
            )
        )
        or bool(
            re.search(
                r"(?:类型为|type\s*[:=]).{0,24}(?:document|web_app|web app|diagram|diff|code_file|image|markdown)",
                text,
                re.IGNORECASE,
            )
        )
        or bool(re.search(r"title\s*(?:为|:|=)", text, re.IGNORECASE))
    )


def get_required_expected_outputs(task: DispatchPlanItem) -> list[DispatchExpectedOutput]:
    return [output for output in task.expectedOutputs or [] if output.required is not False]


# ─── 文本依赖推断 ───────────────────────────────────────────


def _infer_dependencies_for_task(
    task: DispatchPlanItem, previous_tasks: list[DispatchPlanItem]
) -> list[str]:
    inferred: set[str] = set()
    task_text = task.task

    if _has_dependency_signal(task_text):
        for previous in previous_tasks:
            if _contains_task_id_reference(task_text, previous.id):
                inferred.add(previous.id)

    consumed_topics = _get_consumed_artifact_topics(task_text)
    if consumed_topics:
        for previous in previous_tasks:
            produced_topics = _get_produced_artifact_topics(previous.task)
            if consumed_topics & produced_topics:
                inferred.add(previous.id)

    if _is_review_task(task_text):
        for previous in previous_tasks:
            if task_expects_artifact(previous) or _get_produced_artifact_topics(previous.task):
                inferred.add(previous.id)

    # 保持前序任务的相对顺序（与 plan 顺序一致）
    return [previous.id for previous in previous_tasks if previous.id in inferred]


def _has_dependency_signal(text: str) -> bool:
    return bool(
        re.search(
            r"(读取|基于|参考|根据|按照|依赖|等待|待.{0,12}完成|前序|上游|产物|输出|结果|审查|检查|验收|read|review|artifact)",
            text,
            re.IGNORECASE,
        )
    )


def _contains_task_id_reference(text: str, task_id: str) -> bool:
    return bool(
        re.search(
            rf"(^|[^A-Za-z0-9_-]){re.escape(task_id)}([^A-Za-z0-9_-]|$)",
            text,
            re.IGNORECASE,
        )
    )


def _get_consumed_artifact_topics(text: str) -> set[ArtifactTopic]:
    topics: set[ArtifactTopic] = set()
    if _consumes_prd(text):
        topics.add("prd")
    if _consumes_ui_design(text):
        topics.add("ui_design")
    if _consumes_frontend(text):
        topics.add("frontend")
    return topics


def _get_produced_artifact_topics(text: str) -> set[ArtifactTopic]:
    topics: set[ArtifactTopic] = set()
    if _produces_prd(text):
        topics.add("prd")
    if _produces_ui_design(text):
        topics.add("ui_design")
    if _produces_frontend(text):
        topics.add("frontend")
    return topics


def _consumes_prd(text: str) -> bool:
    return bool(
        re.search(
            r"(?:读取|基于|参考|根据|按照|了解|审查|检查|验收|read|review).{0,40}(?:PRD|产品需求|需求文档)"
            r"|(?:PRD|产品需求|需求文档).{0,40}(?:读取|基于|参考|根据|按照|了解|审查|检查|验收|符合|read|review)",
            text,
            re.IGNORECASE,
        )
    )


def _consumes_ui_design(text: str) -> bool:
    return bool(
        re.search(
            r"(?:读取|基于|参考|根据|按照|了解|审查|检查|验收|read|review).{0,40}(?:UI|设计稿|设计方案|风格指南)"
            r"|(?:UI|设计稿|设计方案|风格指南).{0,40}(?:读取|基于|参考|根据|按照|了解|审查|检查|验收|符合|read|review)",
            text,
            re.IGNORECASE,
        )
    )


def _consumes_frontend(text: str) -> bool:
    return bool(
        re.search(
            r"(?:读取|基于|参考|根据|按照|了解|审查|检查|验收|read|review).{0,48}(?:前端|web_app|web app|HTML|网页|实现|代码)"
            r"|(?:前端|web_app|web app|HTML|网页|实现|代码).{0,48}"
            r"(?:读取|基于|参考|根据|按照|了解|审查|检查|验收|符合|产出|artifact|read|review)",
            text,
            re.IGNORECASE,
        )
    )


def _produces_prd(text: str) -> bool:
    return bool(
        re.search(
            r"(?:产出|输出|撰写|写入|生成|创建).{0,32}(?:PRD|产品需求|需求文档)"
            r"|(?:PRD|产品需求|需求文档).{0,32}(?:产出|输出|撰写|写入|生成|创建)",
            text,
            re.IGNORECASE,
        )
    )


def _produces_ui_design(text: str) -> bool:
    return bool(
        re.search(
            r"(?:产出|输出|设计|写入|生成|创建).{0,32}(?:UI|设计稿|设计方案|风格指南)"
            r"|(?:UI|设计稿|设计方案|风格指南).{0,32}(?:产出|输出|写入|生成|创建)",
            text,
            re.IGNORECASE,
        )
    )


def _produces_frontend(text: str) -> bool:
    return bool(
        re.search(
            r"(?:实现|开发|输出|产出|写入|生成|创建).{0,48}(?:前端|web_app|web app|HTML|网页|代码|应用)"
            r"|(?:前端|web_app|web app|HTML|网页|代码|应用).{0,48}(?:实现|开发|输出|产出|写入|生成|创建)",
            text,
            re.IGNORECASE,
        )
    )


def _is_review_task(text: str) -> bool:
    return bool(re.search(r"审查|检查|验收|review|inspect|validate", text, re.IGNORECASE))


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


# ─── 动态重规划（dynamic re-planning）────────────────────────


def should_replan(views: list[ReplanTaskView], conflicts: list[ReplanConflictView]) -> bool:
    """本轮执行后是否需要 Orchestrator 再 plan 补救：有非 complete 任务，或有写冲突。"""
    return any(view.status != "complete" for view in views) or bool(conflicts)


def build_replan_context(
    views: list[ReplanTaskView], conflicts: list[ReplanConflictView]
) -> str:
    """上一轮结果（已完成 / 失败 / 冲突）+ 补救指示 → 补救轮 plan 阶段的 user prompt 前缀。"""
    done = [v for v in views if v.status == "complete"]
    failed = [v for v in views if v.status != "complete"]
    lines = ["<previous_round_results>"]
    for v in done:
        lines.append(f'  <task id="{v.taskId}" agent="{v.agentId}" status="complete" />')
    for v in failed:
        err = f" error={json.dumps(v.error)}" if v.error else ""
        lines.append(f'  <task id="{v.taskId}" agent="{v.agentId}" status="{v.status}"{err} />')
    lines.append("</previous_round_results>")
    if conflicts:
        lines.append("<file_conflicts>")
        for c in conflicts:
            lines.append(
                f'  <conflict path={json.dumps(c.path)} '
                f'tasks={json.dumps(", ".join(c.taskIds))} />'
            )
        lines.append("</file_conflicts>")
    lines.append("")
    lines.append(
        "上一轮存在未完成任务或写冲突。请围绕 original_request 的原始目标输出补救 plan_tasks，"
        "只修复未完成 / 冲突 / 缺失证据的部分：可换更合适的 agent、把写同一文件的任务用 dependsOn "
        "串行化、或把任务拆得更细。不要把实现任务缩小成静态审查、总结或解释；除非用户明确同意缩小范围，"
        "否则补救计划必须继续追踪原始目标的未完成验收。已 complete 的任务不要重做；补救任务需要基于已 "
        "complete 任务时，可以在 dependsOn / inputs 中引用上一轮的 task id，系统会把它当作已解析的外部依赖。"
        "若判断无需或无法补救，就不要调用 plan_tasks（直接进入总结）。"
    )
    return "\n".join(lines)


def build_revise_context(current_plan: list[DispatchPlanItem], feedback: str) -> str:
    """待审计划 + 用户自然语言修改意见 → 重规划轮的 user prompt 前缀。"""
    lines = ["<current_plan>"]
    for task in current_plan:
        deps = (
            f' dependsOn={json.dumps(", ".join(task.dependsOn))}'
            if task.dependsOn
            else ""
        )
        lines.append(f'  <task id="{task.id}" agent="{task.agentId}"{deps}>{task.task}</task>')
    lines.append("</current_plan>")
    lines.extend(
        [
            "<user_revision_request>",
            feedback,
            "</user_revision_request>",
            "",
            "用户对上面这份待执行计划提出了修改意见。请据此调整，重新调用 plan_tasks 输出**完整的新计划**："
            "保留未被要求改动的任务，只改动用户要求的部分（依赖、执行者、任务描述、拆分等）。",
        ]
    )
    return "\n".join(lines)
