"""Orchestrator 同波次「代码冲突」检测的写入追踪。

记录每个子 run 经 fs_write 写过的 workspace 文件（绝对路径 → 内容 hash），
供调度器在一波并行子任务结束后检测「多个子 agent 写了同一文件」。

盲区：bash 工具与平台自带写盘能力不经过 fs_write，不在此记录，这类并发写不检测。
检测到冲突只上报（写进补救上下文与聚合消息），**不做**自动合并。
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict

_writes_by_run: dict[str, dict[str, str]] = {}


class RunFileWrites(BaseModel):
    """一个子 run 的写入集合（绝对路径 → 内容 sha1）。"""

    model_config = ConfigDict(extra="ignore")
    taskId: str
    agentId: str
    runId: str
    writes: dict[str, str]


class FileWriteContributor(BaseModel):
    model_config = ConfigDict(extra="ignore")
    taskId: str
    agentId: str
    runId: str


class FileWriteConflict(BaseModel):
    """同一文件被多个子 run 以不同内容写过。"""

    model_config = ConfigDict(extra="ignore")
    path: str
    contributors: list[FileWriteContributor]


def record_file_write(run_id: str, absolute_path: str, content: str) -> None:
    files = _writes_by_run.setdefault(run_id, {})
    files[absolute_path] = hashlib.sha1(content.encode("utf-8")).hexdigest()


def get_file_writes(run_id: str) -> dict[str, str]:
    return _writes_by_run.get(run_id) or {}


def clear_file_writes(run_id: str) -> None:
    _writes_by_run.pop(run_id, None)


def detect_wave_conflicts(runs: list[RunFileWrites]) -> list[FileWriteConflict]:
    """检测同一波并行子任务的写冲突：≥2 个子 run 写了同一文件且内容不同（hash 不同）。

    内容相同的并发写不算冲突（两个 agent 恰好写出一样的东西）。纯函数，便于单测。
    """
    by_path: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for abs_path, file_hash in run.writes.items():
            writers = by_path.setdefault(abs_path, [])
            writers.append(
                {
                    "taskId": run.taskId,
                    "agentId": run.agentId,
                    "runId": run.runId,
                    "hash": file_hash,
                }
            )

    conflicts: list[FileWriteConflict] = []
    for abs_path, writers in by_path.items():
        if len(writers) < 2:
            continue
        if len({writer["hash"] for writer in writers}) < 2:
            continue
        conflicts.append(
            FileWriteConflict(
                path=abs_path,
                contributors=[
                    FileWriteContributor(
                        taskId=writer["taskId"],
                        agentId=writer["agentId"],
                        runId=writer["runId"],
                    )
                    for writer in writers
                ],
            )
        )
    return conflicts
