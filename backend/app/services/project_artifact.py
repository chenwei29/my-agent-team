"""project 产物：从 run 的文件写入证据生成文件清单，并落一条 project 类型 artifact。

代码类子任务不走 write_artifact —— 子 Agent 只管往 workspace 写文件，
系统在任务收尾时把本轮写过的文件打包成 project 产物（正文留在 workspace，
DB 只存清单），供下游任务与用户查看。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db.models import Agent, Artifact, Workspace
from app.db.session import SessionLocal
from app.schemas.artifacts import ProjectFile
from app.schemas.dispatch import RunFileEvidence
from app.schemas.events import ArtifactCreateEvent, ArtifactRecord
from app.security.workspace_utils import get_effective_cwd, is_path_within
from app.services.event_bus import event_bus
from app.utils.ids import new_artifact_id
from app.utils.time import now_ms


def build_project_files(
    file_writes: list[RunFileEvidence], workspace_root: str
) -> list[ProjectFile]:
    """写入证据 → 相对路径文件清单（去重、按路径排序）。

    absolutePath 是可信字段（工具入参的 path 可能相对也可能绝对）；
    落在 workspace 之外的记录直接丢弃。
    """
    by_path: dict[str, ProjectFile] = {}
    for file_write in file_writes:
        if not is_path_within(file_write.absolutePath, workspace_root):
            continue
        rel = _to_rel(file_write.absolutePath, workspace_root)
        if not rel:
            continue
        previous = by_path.get(rel)
        size = (
            file_write.bytes
            if file_write.bytes is not None
            else (previous.size_bytes if previous else 0)
        )
        by_path[rel] = ProjectFile(path=rel, size_bytes=size)
    return sorted(by_path.values(), key=lambda item: item.path)


def _to_rel(abs_path: str, root: str) -> str | None:
    try:
        rel = Path(abs_path).relative_to(root)
    except ValueError:
        return None
    text = rel.as_posix()
    if not text or text == ".":
        return None
    return text


def normalize_project_path(input_path: str) -> str | None:
    """清单里的路径再校验一次：拒绝绝对路径与 `..` 逃逸。"""
    if not input_path or os.path.isabs(input_path) or (len(input_path) > 1 and input_path[1] == ":"):
        return None
    parts = [part for part in input_path.replace("\\", "/").split("/") if part and part != "."]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


async def maybe_create_project_artifact(
    *,
    evidence_file_writes: list[RunFileEvidence],
    conversation_id: str,
    agent_id: str,
    task_id: str | None = None,
) -> str | None:
    """本轮有 workspace 写入时生成 project artifact，返回 artifact id（无写入返回 None）。

    只负责创建 + 广播 artifact.create；结果集里怎么登记由调用方决定
    （子任务路径由调度器挂到 DispatchTaskResult，普通 run 由 runner 挂到执行结果）。
    """
    if not evidence_file_writes:
        return None

    async with SessionLocal() as session:
        workspace = await session.scalar(
            select(Workspace).where(Workspace.conversation_id == conversation_id)
        )
        if workspace is None:
            return None
        files = build_project_files(evidence_file_writes, get_effective_cwd(workspace))
        if not files:
            return None

        agent = await session.scalar(select(Agent).where(Agent.id == agent_id))
        title = f"{agent.name if agent else agent_id} · 项目产物"

        content: dict[str, Any] = {
            "type": "project",
            "files": [file.model_dump(by_alias=True) for file in files],
            "agentId": agent_id,
        }
        if task_id:
            content["taskId"] = task_id

        artifact_id = new_artifact_id()
        created_at = now_ms()
        session.add(
            Artifact(
                id=artifact_id,
                conversation_id=conversation_id,
                type="project",
                title=title,
                content=content,
                version=1,
                parent_artifact_id=None,
                created_by_agent_id=agent_id,
                created_at=created_at,
            )
        )
        await session.commit()

    event_bus.publish(
        ArtifactCreateEvent(
            conversationId=conversation_id,
            timestamp=created_at,
            artifact=ArtifactRecord(
                id=artifact_id,
                conversationId=conversation_id,
                type="project",
                title=title,
                content=content,
                version=1,
                parentArtifactId=None,
                createdByAgentId=agent_id,
                createdAt=created_at,
            ),
        )
    )
    return artifact_id
