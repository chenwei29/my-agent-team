"""按 run 收集的工具证据（fs_write 落盘 / 命令执行），供任务证据门禁判定。

每个子 run 一份：fs_write 落盘成功、bash（含 requiredCommands 补跑）执行完，
都往这里追加一条记录；run 结束后由调度器取走并清理。进程内存存储，不落库。
"""

from __future__ import annotations

from app.schemas.dispatch import (
    RunCommandEvidence,
    RunFileEvidence,
    RunToolEvidence,
)

_evidence_by_run: dict[str, RunToolEvidence] = {}


def _ensure_evidence(run_id: str) -> RunToolEvidence:
    evidence = _evidence_by_run.get(run_id)
    if evidence is None:
        evidence = RunToolEvidence()
        _evidence_by_run[run_id] = evidence
    return evidence


def record_run_file_write(run_id: str, evidence: RunFileEvidence) -> None:
    _ensure_evidence(run_id).fileWrites.append(evidence)


def record_run_command(run_id: str, evidence: RunCommandEvidence) -> None:
    _ensure_evidence(run_id).commands.append(evidence)


def get_run_tool_evidence(run_id: str) -> RunToolEvidence:
    return _evidence_by_run.get(run_id) or RunToolEvidence()


def clear_run_tool_evidence(run_id: str) -> None:
    _evidence_by_run.pop(run_id, None)
