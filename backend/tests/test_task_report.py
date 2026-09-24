"""子任务结果上报与证据门禁测试。

错误文案与判定顺序是调度器对子任务的完成契约：文案被原样写进 dispatch.end.error
与聚合消息，判定顺序决定「哪条门禁先拦」，改动会直接影响用户看到的失败原因。
"""

from __future__ import annotations

import json

from app.schemas.dispatch import (
    DispatchPlanItem,
    RunCommandEvidence,
    RunFileEvidence,
    RunToolEvidence,
    TaskResultReport,
)
from app.services.task_result_report import (
    dump_report,
    evaluate_task_result_report,
    is_task_result_report_tool_name,
    read_task_result_report_from_tool_result,
)


def task(**overrides) -> DispatchPlanItem:
    base = {
        "id": "t1",
        "agentId": "ag_reviewer",
        "task": "Review the implementation",
    }
    base.update(overrides)
    return DispatchPlanItem(**base)


complete_report_dict = {
    "status": "complete",
    "summary": "Reviewed the implementation and found it acceptable.",
    "acceptanceResults": [
        {
            "criterion": "Checks PRD alignment",
            "passed": True,
            "evidence": "The implementation covers the required PRD scope.",
        },
    ],
}
completeReport = TaskResultReport(**complete_report_dict)


class TestReadTaskResultReportFromToolResult:
    def test_parses_direct_custom_agent_tool_results(self):
        assert dump_report(read_task_result_report_from_tool_result(completeReport)) == (
            complete_report_dict
        )

    def test_parses_claude_mcp_text_content(self):
        assert dump_report(read_task_result_report_from_tool_result(
            [{"type": "text", "text": json.dumps(complete_report_dict)}]
        )) == complete_report_dict

    def test_parses_codex_mcp_wrapper_results(self):
        assert dump_report(read_task_result_report_from_tool_result({
            "result": {"structuredContent": complete_report_dict},
            "status": "completed",
        })) == complete_report_dict


class TestIsTaskResultReportToolName:
    def test_matches_direct_and_mcp_prefixed_tool_names(self):
        assert is_task_result_report_tool_name("report_task_result") is True
        assert is_task_result_report_tool_name("mcp__agenthub__report_task_result") is True
        assert is_task_result_report_tool_name("codex_mcp_agenthub_report_task_result") is True
        assert is_task_result_report_tool_name("write_artifact") is False


class TestEvaluateTaskResultReport:
    def test_accepts_complete_reports_with_matching_acceptance_criteria(self):
        assert evaluate_task_result_report(
            task(acceptanceCriteria=["Checks PRD alignment"]),
            completeReport,
        ) == {"ok": True}

    def test_does_not_use_expected_outputs_as_a_completion_gate(self):
        assert evaluate_task_result_report(
            task(expectedOutputs=[{"id": "report", "type": "document"}]),
            TaskResultReport(
                status="complete",
                summary="The review was completed in the final message.",
            ),
        ) == {"ok": True}

    def test_fails_when_a_child_task_omits_report_task_result(self):
        assert evaluate_task_result_report(task(), None) == {
            "ok": False,
            "error": 'Task "t1" completed without report_task_result',
        }

    def test_fails_when_the_child_reports_failed_or_blocked(self):
        assert evaluate_task_result_report(task(), TaskResultReport(
            status="blocked",
            summary="Need missing credentials.",
            blockers=["Missing API key"],
        )) == {
            "ok": False,
            "error": "Task \"t1\" reported blocked: Need missing credentials. Blockers: Missing API key",
        }

    def test_fails_when_acceptance_criteria_are_missing_or_failed(self):
        assert evaluate_task_result_report(
            task(acceptanceCriteria=["Checks PRD alignment"]),
            TaskResultReport(status="complete", summary="Done."),
        ) == {
            "ok": False,
            "error": 'Task "t1" report is missing acceptance criteria result(s): Checks PRD alignment',
        }

        assert evaluate_task_result_report(task(), TaskResultReport(
            status="complete",
            summary="Done.",
            acceptanceResults=[
                {
                    "criterion": "Checks PRD alignment",
                    "passed": False,
                    "evidence": "The implementation missed the export workflow.",
                },
            ],
        )) == {
            "ok": False,
            "error": (
                'Task "t1" did not satisfy acceptance criteria: Checks PRD alignment '
                "(The implementation missed the export workflow.)"
            ),
        }

    def test_accepts_complete_reports_with_required_file_and_command_evidence(self):
        assert evaluate_task_result_report(
            task(
                targetPaths=["src/foo.ts"],
                requiredCommands=[{"command": "pnpm test src/foo.test.ts"}],
                requiredEvidence=["测试命令 exitCode=0"],
            ),
            TaskResultReport(**{
                "status": "complete",
                "summary": "Implemented foo. 测试命令 exitCode=0",
                "filesChanged": [{"path": "src/foo.ts", "action": "modified"}],
                "commandsRun": [{"command": "pnpm test src/foo.test.ts", "exitCode": 0}],
            }),
            RunToolEvidence(
                fileWrites=[
                    RunFileEvidence(
                        path="src/foo.ts",
                        absolutePath="E:/repo/src/foo.ts",
                        bytes=123,
                        applied="auto",
                    ),
                ],
                commands=[
                    RunCommandEvidence(
                        command="pnpm test src/foo.test.ts",
                        cwd="E:/repo",
                        exitCode=0,
                        timedOut=False,
                        isError=False,
                    ),
                ],
            ),
        ) == {"ok": True}

    def test_fails_complete_reports_missing_required_file_or_command_evidence(self):
        assert evaluate_task_result_report(
            task(
                targetPaths=["src/foo.ts"],
                requiredCommands=[{"command": "pnpm test src/foo.test.ts"}],
            ),
            TaskResultReport(status="complete", summary="Done."),
        ) == {
            "ok": False,
            "error": 'Task "t1" report is missing target path evidence: src/foo.ts',
        }

        assert evaluate_task_result_report(
            task(requiredCommands=[{"command": "pnpm test src/foo.test.ts"}]),
            TaskResultReport(status="complete", summary="Done."),
        ) == {
            "ok": False,
            "error": (
                'Task "t1" report is missing successful command evidence: '
                "pnpm test src/foo.test.ts"
            ),
        }

    def test_fails_complete_reports_when_managed_command_evidence_failed(self):
        assert evaluate_task_result_report(
            task(),
            TaskResultReport(status="complete", summary="Done."),
            RunToolEvidence(
                fileWrites=[],
                commands=[
                    RunCommandEvidence(
                        command="mvn compile",
                        cwd="E:/repo",
                        exitCode=1,
                        timedOut=False,
                        isError=False,
                    ),
                ],
            ),
        ) == {
            "ok": False,
            "error": 'Task "t1" has failed command evidence: mvn compile (exit 1)',
        }

    def test_accepts_when_a_failed_managed_command_later_succeeds(self):
        assert evaluate_task_result_report(
            task(requiredCommands=[{"command": "pnpm build", "cwd": "frontend"}]),
            TaskResultReport(**{
                "status": "complete",
                "summary": "Build now passes.",
                "commandsRun": [{"command": "pnpm build", "exitCode": 0, "cwd": "frontend"}],
            }),
            RunToolEvidence(
                fileWrites=[],
                commands=[
                    RunCommandEvidence(
                        command="pnpm build",
                        cwd="E:/repo/frontend",
                        exitCode=1,
                        timedOut=False,
                        isError=False,
                    ),
                    RunCommandEvidence(
                        command="pnpm build",
                        cwd="E:/repo/frontend",
                        exitCode=0,
                        timedOut=False,
                        isError=False,
                    ),
                ],
            ),
        ) == {"ok": True}

    def test_does_not_let_an_earlier_automatic_prepare_failure_block_later_successful_verification(
        self,
    ):
        assert evaluate_task_result_report(
            task(requiredCommands=[{"command": "pnpm build", "cwd": "frontend"}]),
            TaskResultReport(**{
                "status": "complete",
                "summary": "Build passes after dependencies were prepared.",
                "commandsRun": [{"command": "pnpm build", "exitCode": 0, "cwd": "frontend"}],
            }),
            RunToolEvidence(
                fileWrites=[],
                commands=[
                    RunCommandEvidence(
                        command="pnpm install",
                        cwd="E:/repo/frontend",
                        exitCode=1,
                        timedOut=False,
                        isError=False,
                        prepare=True,
                    ),
                    RunCommandEvidence(
                        command="pnpm build",
                        cwd="E:/repo/frontend",
                        exitCode=1,
                        timedOut=False,
                        isError=True,
                        error="prepare command failed",
                    ),
                    RunCommandEvidence(
                        command="pnpm build",
                        cwd="E:/repo/frontend",
                        exitCode=0,
                        timedOut=False,
                        isError=False,
                    ),
                ],
            ),
        ) == {"ok": True}

    def test_fails_code_tasks_without_successful_runnable_verification_command_evidence(self):
        assert evaluate_task_result_report(
            task(task="Implement the frontend app", taskKind="code"),
            TaskResultReport(status="complete", summary="Implemented the app."),
            RunToolEvidence(
                fileWrites=[
                    RunFileEvidence(
                        path="frontend/src/App.tsx",
                        absolutePath="E:/repo/frontend/src/App.tsx",
                        bytes=123,
                        applied="auto",
                    ),
                ],
                commands=[],
            ),
        ) == {
            "ok": False,
            "error": (
                'Task "t1" is missing successful runnable verification command evidence: '
                "build/compile/test/typecheck/lint command exitCode=0"
            ),
        }

    def test_does_not_count_prepare_commands_as_runnable_verification(self):
        assert evaluate_task_result_report(
            task(task="Implement the frontend app", taskKind="code"),
            TaskResultReport(
                status="complete", summary="Dependencies installed and app implemented."
            ),
            RunToolEvidence(
                fileWrites=[],
                commands=[
                    RunCommandEvidence(
                        command="pnpm install",
                        cwd="E:/repo/frontend",
                        exitCode=0,
                        timedOut=False,
                        isError=False,
                        prepare=True,
                    ),
                ],
            ),
        ) == {
            "ok": False,
            "error": (
                'Task "t1" is missing successful runnable verification command evidence: '
                "build/compile/test/typecheck/lint command exitCode=0"
            ),
        }

    def test_accepts_code_tasks_with_successful_build_command_evidence(self):
        assert evaluate_task_result_report(
            task(task="Implement the frontend app", taskKind="code"),
            TaskResultReport(status="complete", summary="Build passes."),
            RunToolEvidence(
                fileWrites=[],
                commands=[
                    RunCommandEvidence(
                        command="pnpm build",
                        cwd="E:/repo/frontend",
                        exitCode=0,
                        timedOut=False,
                        isError=False,
                    ),
                ],
            ),
        ) == {"ok": True}

    def test_does_not_count_unrelated_successful_commands_as_runnable_verification(self):
        assert evaluate_task_result_report(
            task(task="Implement the frontend app", taskKind="code"),
            TaskResultReport(status="complete", summary="Listed files."),
            RunToolEvidence(
                fileWrites=[],
                commands=[
                    RunCommandEvidence(
                        command="ls",
                        cwd="E:/repo/frontend",
                        exitCode=0,
                        timedOut=False,
                        isError=False,
                    ),
                ],
            ),
        ) == {
            "ok": False,
            "error": (
                'Task "t1" is missing successful runnable verification command evidence: '
                "build/compile/test/typecheck/lint command exitCode=0"
            ),
        }

    def test_allows_non_code_review_tasks_to_complete_without_runnable_verification(self):
        assert evaluate_task_result_report(
            task(task="Review the implementation", taskKind="review"),
            TaskResultReport(status="complete", summary="Reviewed the implementation."),
        ) == {"ok": True}
