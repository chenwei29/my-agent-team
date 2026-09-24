"""子任务结果上报（report_task_result）的解析与证据门禁（纯函数模块）。

子 Agent 收尾时上报结构化结果；调度器用这里的判定决定任务是否真的完成：

- `read_task_result_report_from_tool_result`：从各种适配器的 tool result 包装形态里
  （直接对象 / JSON 字符串 / content 数组 / structuredContent 包装）递归抠出上报；
- `evaluate_task_result_report`：证据门禁 —— 有序判定「没上报 / 上报失败 / 命令失败 /
  验收不过 / 缺验收结果 / 缺目标路径证据 / 缺命令证据 / 代码任务缺可运行验证 / 缺证据」，
  错误文案带任务定位，会被原样写进 dispatch.end.error 与聚合消息。

判定用的运行期证据（fs_write 落盘 / 命令执行）由 dispatch_run_evidence 按 run 收集。
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas.dispatch import (
    DispatchPlanItem,
    RunCommandEvidence,
    RunToolEvidence,
    TaskAcceptanceResult,
    TaskCommandEvidence,
    TaskFileEvidence,
    TaskResultReport,
    TaskResultReportStatus,
    TaskTestEvidence,
)
from app.services.dispatch_plan import (
    CODE_TASK_RUNNABLE_REQUIRED_EVIDENCE,
    is_code_implementation_task,
)

REPORT_TASK_RESULT_TOOL_NAME = "report_task_result"


class TaskResultReportEvaluation(TypedDict, total=False):
    ok: bool
    error: str


# ─── 入参模型（LLM 工具调用的 args）──────────────────────────
class _AcceptanceResultArg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    criterion: str = Field(min_length=1)
    passed: bool
    evidence: str = Field(min_length=1)


class _FileChangedArg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(min_length=1)
    action: str | None = None  # 'created' | 'modified' | 'deleted' | 'verified'


class _CommandRunArg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    command: str = Field(min_length=1)
    exitCode: int | None
    cwd: str | None = None
    timedOut: bool | None = None
    summary: str | None = None


class _TestArg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    command: str = Field(min_length=1)
    passed: bool
    summary: str | None = None


class ReportTaskResultArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: TaskResultReportStatus
    summary: str = Field(min_length=1)
    acceptanceResults: list[_AcceptanceResultArg] | None = None
    filesChanged: list[_FileChangedArg] | None = None
    commandsRun: list[_CommandRunArg] | None = None
    tests: list[_TestArg] | None = None
    blockers: list[str] | None = None


def normalize_task_result_report(data: ReportTaskResultArgs) -> TaskResultReport:
    """裁掉空白项、省略空数组，得到上报的规范形态。"""
    acceptance_results = None
    if data.acceptanceResults is not None:
        acceptance_results = [
            TaskAcceptanceResult(
                criterion=result.criterion.strip(),
                passed=result.passed,
                evidence=result.evidence.strip(),
            )
            for result in data.acceptanceResults
        ]
        acceptance_results = [
            result for result in acceptance_results if result.criterion and result.evidence
        ]

    blockers = None
    if data.blockers is not None:
        blockers = [blocker.strip() for blocker in data.blockers if blocker.strip()]

    files_changed = None
    if data.filesChanged is not None:
        files_changed = [
            TaskFileEvidence(path=file.path.strip(), action=file.action)  # type: ignore[arg-type]
            for file in data.filesChanged
            if file.path.strip()
        ]

    commands_run = None
    if data.commandsRun is not None:
        commands_run = [
            TaskCommandEvidence(
                command=command.command.strip(),
                exitCode=command.exitCode,
                cwd=command.cwd.strip() if command.cwd and command.cwd.strip() else None,
                timedOut=command.timedOut,
                summary=command.summary.strip() if command.summary and command.summary.strip() else None,
            )
            for command in data.commandsRun
            if command.command.strip()
        ]

    tests = None
    if data.tests is not None:
        tests = [
            TaskTestEvidence(
                command=test.command.strip(),
                passed=test.passed,
                summary=test.summary.strip() if test.summary and test.summary.strip() else None,
            )
            for test in data.tests
            if test.command.strip()
        ]

    return TaskResultReport(
        status=data.status,
        summary=data.summary.strip(),
        acceptanceResults=acceptance_results or None,
        filesChanged=files_changed or None,
        commandsRun=commands_run or None,
        tests=tests or None,
        blockers=blockers or None,
    )


def dump_report(report: TaskResultReport) -> dict[str, Any]:
    """上报 → 直传 JSON dict：空数组省略，`exitCode: null` 保留。"""
    out: dict[str, Any] = {"status": report.status, "summary": report.summary}
    if report.acceptanceResults:
        out["acceptanceResults"] = [
            {"criterion": r.criterion, "passed": r.passed, "evidence": r.evidence}
            for r in report.acceptanceResults
        ]
    if report.filesChanged:
        out["filesChanged"] = [
            {"path": f.path, **({"action": f.action} if f.action else {})}
            for f in report.filesChanged
        ]
    if report.commandsRun:
        out["commandsRun"] = [
            {
                "command": c.command,
                "exitCode": c.exitCode,
                **({"cwd": c.cwd} if c.cwd else {}),
                **({"timedOut": c.timedOut} if c.timedOut is not None else {}),
                **({"summary": c.summary} if c.summary else {}),
            }
            for c in report.commandsRun
        ]
    if report.tests:
        out["tests"] = [
            {
                "command": t.command,
                "passed": t.passed,
                **({"summary": t.summary} if t.summary else {}),
            }
            for t in report.tests
        ]
    if report.blockers:
        out["blockers"] = list(report.blockers)
    return out


def parse_task_result_report(value: Any) -> TaskResultReport | None:
    if isinstance(value, BaseModel):
        # 已是模型实例（重复解析路径）：先摊成 dict 再走统一校验
        value = value.model_dump()
    try:
        parsed = ReportTaskResultArgs.model_validate(value)
    except (ValidationError, ValueError, TypeError):
        return None
    return normalize_task_result_report(parsed)


def read_task_result_report_from_tool_result(result: Any) -> TaskResultReport | None:
    return _read_task_result_report_from_unknown(result, 0)


def is_task_result_report_tool_name(tool_name: str) -> bool:
    return (
        tool_name == REPORT_TASK_RESULT_TOOL_NAME
        or tool_name.endswith(f"__{REPORT_TASK_RESULT_TOOL_NAME}")
        or tool_name.endswith(f"_{REPORT_TASK_RESULT_TOOL_NAME}")
    )


# ─── 证据门禁 ───────────────────────────────────────────────


def evaluate_task_result_report(
    task: DispatchPlanItem,
    report: TaskResultReport | None,
    evidence: RunToolEvidence | None = None,
) -> TaskResultReportEvaluation:
    """有序判定任务完成门禁；第一条不满足即返回带定位的错误。"""
    if evidence is None:
        evidence = RunToolEvidence()

    if report is None:
        return {"ok": False, "error": f'Task "{task.id}" completed without report_task_result'}

    if report.status != "complete":
        return {"ok": False, "error": _format_reported_non_completion(task.id, report)}

    failed_commands = [
        command
        for index, command in enumerate(evidence.commands)
        if not command.prepare
        and _is_failed_command(command)
        and not _has_later_successful_command(command, index, evidence.commands)
    ]
    if failed_commands:
        return {
            "ok": False,
            "error": f'Task "{task.id}" has failed command evidence: '
            + "; ".join(
                f"{command.command} ("
                + (
                    (command.error or "tool error")
                    if command.isError
                    else ("timed out" if command.timedOut else f"exit {command.exitCode}")
                )
                + ")"
                for command in failed_commands
            ),
        }

    failed_acceptance = [result for result in (report.acceptanceResults or []) if not result.passed]
    if failed_acceptance:
        return {
            "ok": False,
            "error": f'Task "{task.id}" did not satisfy acceptance criteria: '
            + "; ".join(f"{result.criterion} ({result.evidence})" for result in failed_acceptance),
        }

    criteria = task.acceptanceCriteria or []
    if criteria:
        reported_criteria = {
            result.criterion.strip() for result in (report.acceptanceResults or [])
        }
        missing = [c for c in criteria if c.strip() not in reported_criteria]
        if missing:
            return {
                "ok": False,
                "error": (
                    f'Task "{task.id}" report is missing acceptance criteria result(s): '
                    + "; ".join(missing)
                ),
            }

    missing_target_paths = [
        target_path
        for target_path in (task.targetPaths or [])
        if not _has_path_evidence(target_path, report, evidence)
    ]
    if missing_target_paths:
        return {
            "ok": False,
            "error": (
                f'Task "{task.id}" report is missing target path evidence: '
                + "; ".join(missing_target_paths)
            ),
        }

    missing_commands = [
        required.command
        for required in (task.requiredCommands or [])
        if not _has_successful_command_evidence(required.command, report, evidence)
    ]
    if missing_commands:
        return {
            "ok": False,
            "error": (
                f'Task "{task.id}" report is missing successful command evidence: '
                + "; ".join(missing_commands)
            ),
        }

    if is_code_implementation_task(task) and not has_successful_verification_command_evidence(
        evidence
    ):
        return {
            "ok": False,
            "error": (
                f'Task "{task.id}" is missing successful runnable verification command evidence: '
                "build/compile/test/typecheck/lint command exitCode=0"
            ),
        }

    missing_evidence = [
        required
        for required in (task.requiredEvidence or [])
        if not _required_evidence_satisfied(required, report, evidence)
    ]
    if missing_evidence:
        return {
            "ok": False,
            "error": (
                f'Task "{task.id}" report is missing required evidence: '
                + "; ".join(missing_evidence)
            ),
        }

    return {"ok": True}


def has_successful_verification_command_evidence(evidence: RunToolEvidence) -> bool:
    return any(_is_successful_verification_command(c) for c in evidence.commands)


def is_verification_command(command: str) -> bool:
    normalized = _normalize_command(command)
    return (not _is_prepare_command(normalized)) and any(
        pattern.search(normalized) for pattern in VERIFICATION_COMMAND_PATTERNS
    )


def _is_successful_verification_command(command: RunCommandEvidence) -> bool:
    return (
        not command.prepare
        and not command.isError
        and not command.timedOut
        and command.exitCode == 0
        and is_verification_command(command.command)
    )


def _required_evidence_satisfied(
    required: str, report: TaskResultReport, evidence: RunToolEvidence
) -> bool:
    if required.strip() == CODE_TASK_RUNNABLE_REQUIRED_EVIDENCE:
        return has_successful_verification_command_evidence(evidence)
    return _evidence_mentions(required, report, evidence)


def _evidence_mentions(
    required: str, report: TaskResultReport, evidence: RunToolEvidence
) -> bool:
    haystack_parts = [report.summary]
    for result in report.acceptanceResults or []:
        haystack_parts += [result.criterion, result.evidence]
    for file in report.filesChanged or []:
        haystack_parts += [file.path, file.action or ""]
    for command in report.commandsRun or []:
        haystack_parts += [command.command, command.summary or ""]
    for test in report.tests or []:
        haystack_parts += [test.command, test.summary or ""]
    for file in evidence.fileWrites:
        haystack_parts += [file.path, file.absolutePath, str(file.bytes) if file.bytes is not None else ""]
    for command in evidence.commands:
        haystack_parts += [
            command.command,
            command.cwd,
            str(command.exitCode) if command.exitCode is not None else "",
            "timedOut" if command.timedOut else "",
            "isError" if command.isError else "",
            command.error or "",
            "exitCode=0"
            if command.exitCode == 0 and not command.timedOut and not command.isError
            else "",
        ]
    return required.lower() in "\n".join(haystack_parts).lower()


def _is_failed_command(command: RunCommandEvidence) -> bool:
    # exitCode 为 None（拿不到退出码）也算失败
    return bool(command.isError or command.timedOut or command.exitCode != 0)


def _has_later_successful_command(
    failed: RunCommandEvidence, failed_index: int, commands: list[RunCommandEvidence]
) -> bool:
    return any(
        _commands_match(failed.command, command.command)
        and not command.isError
        and not command.timedOut
        and command.exitCode == 0
        for command in commands[failed_index + 1 :]
    )


def _has_path_evidence(
    target_path: str, report: TaskResultReport, evidence: RunToolEvidence
) -> bool:
    candidates = [file.path for file in (report.filesChanged or [])]
    for file in evidence.fileWrites:
        candidates += [file.path, file.absolutePath]
    return any(_paths_match(target_path, candidate) for candidate in candidates)


def _has_successful_command_evidence(
    required_command: str, report: TaskResultReport, evidence: RunToolEvidence
) -> bool:
    reported = any(
        _commands_match(required_command, command.command) and command.exitCode == 0
        for command in (report.commandsRun or [])
    )
    tested = any(
        _commands_match(required_command, test.command) and test.passed
        for test in (report.tests or [])
    )
    recorded = any(
        _commands_match(required_command, command.command)
        and not command.isError
        and not command.timedOut
        and command.exitCode == 0
        for command in evidence.commands
    )
    return bool(reported or tested or recorded)


def _paths_match(expected: str, actual: str) -> bool:
    e = _normalize_path(expected)
    a = _normalize_path(actual)
    return a == e or a.endswith(f"/{e}") or a.startswith(f"{e}/")


def _commands_match(expected: str, actual: str) -> bool:
    e = _normalize_command(expected)
    a = _normalize_command(actual)
    return a == e or e in a


def _normalize_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    normalized = re.sub(r"^\./+", "", normalized)
    normalized = re.sub(r"/+$", "", normalized)
    return normalized.lower()


def _normalize_command(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


VERIFICATION_COMMAND_PATTERNS = [
    re.compile(
        r"\b(?:pnpm|npm|yarn|bun)(?:\.cmd)?\b(?=.*\b(?:run\s+)?(?:build|test|lint|typecheck|check|compile)(?:\b|:))",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:tsc|tsc\.cmd)\b", re.IGNORECASE),
    re.compile(r"\bnext(?:\.cmd)?\s+build\b", re.IGNORECASE),
    re.compile(r"\bvite(?:\.cmd)?\s+build\b", re.IGNORECASE),
    re.compile(r"\bmvn(?:\.cmd)?\b(?=.*\b(?:compile|test|package|verify)\b)", re.IGNORECASE),
    re.compile(
        r"\b(?:gradle|gradlew|gradlew\.bat|\.\/gradlew)\b(?=.*\b(?:build|test|check)\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\bgo\s+(?:test|build)\b", re.IGNORECASE),
    re.compile(r"\bcargo\s+(?:test|build|check)\b", re.IGNORECASE),
    re.compile(r"\b(?:pytest|py\.test)\b", re.IGNORECASE),
    re.compile(r"\bpython(?:3)?(?:\.exe)?\s+-m\s+pytest\b", re.IGNORECASE),
    re.compile(r"\bruff\s+check\b", re.IGNORECASE),
    re.compile(r"\bmypy\b", re.IGNORECASE),
    re.compile(r"\bdotnet\s+(?:build|test)\b", re.IGNORECASE),
]


def _is_prepare_command(command: str) -> bool:
    return bool(
        re.match(
            r"\s*(?:pnpm|npm|yarn|bun)(?:\.cmd)?\s+(?:install|i|ci|add)\b", command, re.IGNORECASE
        )
    ) and not bool(
        re.search(r"\b(?:build|test|lint|typecheck|check|compile)(?:\b|:)", command, re.IGNORECASE)
    )


def _read_task_result_report_from_unknown(value: Any, depth: int) -> TaskResultReport | None:
    if depth > 6:
        return None

    direct = parse_task_result_report(value)
    if direct is not None:
        return direct

    if isinstance(value, str):
        return _read_task_result_report_from_json_text(value, depth + 1)

    if isinstance(value, list):
        for item in value:
            parsed = _read_task_result_report_from_unknown(item, depth + 1)
            if parsed is not None:
                return parsed
        return None

    if not isinstance(value, dict):
        return None

    if isinstance(value.get("text"), str):
        parsed = _read_task_result_report_from_json_text(value["text"], depth + 1)
        if parsed is not None:
            return parsed

    for key in ("structuredContent", "structured_content", "result", "value", "content"):
        if key not in value:
            continue
        parsed = _read_task_result_report_from_unknown(value[key], depth + 1)
        if parsed is not None:
            return parsed

    return None


def _read_task_result_report_from_json_text(text: str, depth: int) -> TaskResultReport | None:
    try:
        return _read_task_result_report_from_unknown(json.loads(text), depth)
    except (json.JSONDecodeError, ValueError):
        return None


def _format_reported_non_completion(task_id: str, report: TaskResultReport) -> str:
    blockers = f" Blockers: {'; '.join(report.blockers)}" if report.blockers else ""
    return f'Task "{task_id}" reported {report.status}: {report.summary}{blockers}'
