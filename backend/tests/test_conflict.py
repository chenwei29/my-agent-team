"""同波次写冲突检测测试。

冲突定义是行为契约：≥2 个子 run 写同一绝对路径且内容 hash 不同才算冲突，
相同内容的并发写不报。检测结果只上报不合并。
"""

from __future__ import annotations

from app.services.dispatch_file_writes import RunFileWrites, detect_wave_conflicts


def run(task_id: str, agent_id: str, writes: dict[str, str]) -> RunFileWrites:
    return RunFileWrites(
        taskId=task_id,
        agentId=agent_id,
        runId=f"run_{task_id}",
        writes=writes,
    )


class TestDetectWaveConflicts:
    def test_returns_no_conflict_when_runs_touch_different_files(self):
        assert detect_wave_conflicts([
            run("t1", "ag_pm", {"/ws/a.md": "h1"}),
            run("t2", "ag_fe", {"/ws/b.ts": "h2"}),
        ]) == []

    def test_flags_two_runs_writing_the_same_file_with_different_content(self):
        conflicts = detect_wave_conflicts([
            run("t1", "ag_fe", {"/ws/index.html": "hashA"}),
            run("t2", "ag_design", {"/ws/index.html": "hashB"}),
        ])
        assert len(conflicts) == 1
        assert conflicts[0].path == "/ws/index.html"
        assert sorted(c.taskId for c in conflicts[0].contributors) == ["t1", "t2"]

    def test_does_not_flag_identical_concurrent_writes_same_hash(self):
        assert detect_wave_conflicts([
            run("t1", "ag_fe", {"/ws/index.html": "same"}),
            run("t2", "ag_design", {"/ws/index.html": "same"}),
        ]) == []

    def test_detects_a_conflict_among_three_writers_and_lists_all_contributors(self):
        conflicts = detect_wave_conflicts([
            run("t1", "a", {"/ws/x": "h1"}),
            run("t2", "b", {"/ws/x": "h2"}),
            run("t3", "c", {"/ws/x": "h1"}),
        ])
        assert len(conflicts) == 1
        assert len(conflicts[0].contributors) == 3

    def test_ignores_a_single_run_even_if_it_writes_many_files(self):
        assert detect_wave_conflicts([
            run("t1", "a", {"/ws/a": "h1", "/ws/b": "h2", "/ws/c": "h3"}),
        ]) == []
