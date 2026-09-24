"""产物领域逻辑：全局列表（带会话标题）、删除、用户面板提交新版本。

列表一次 JOIN 出会话标题，避免前端 N+1；新版本继承 parent 的
conversationId / type / createdByAgentId（免迁移、免 FK 问题），
version = parent.version + 1，内容校验与 write_artifact 共用
build_artifact_content（单一来源）。
"""

from __future__ import annotations

from sqlalchemy import delete as sa_delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Artifact, Conversation
from app.errors import NotFoundError, ServiceError
from app.services.artifact_content import build_artifact_content, describe_artifact_content_error
from app.utils.ids import new_artifact_id
from app.utils.time import now_ms


async def list_artifacts(session: AsyncSession) -> list[dict]:
    rows = (
        await session.execute(select(Artifact).order_by(desc(Artifact.created_at)))
    ).scalars().all()
    if not rows:
        return []

    conv_ids = {row.conversation_id for row in rows}
    convs = (
        await session.execute(select(Conversation).where(Conversation.id.in_(conv_ids)))
    ).scalars().all()
    title_by_id = {conv.id: conv.title for conv in convs}

    return [
        {
            "id": row.id,
            "conversationId": row.conversation_id,
            "conversationTitle": title_by_id.get(row.conversation_id),
            "type": row.type,
            "title": row.title,
            "version": row.version,
            "parentArtifactId": row.parent_artifact_id,
            "createdByAgentId": row.created_by_agent_id,
            "createdAt": row.created_at,
        }
        for row in rows
    ]


async def get_artifact(session: AsyncSession, artifact_id: str) -> Artifact | None:
    return await session.get(Artifact, artifact_id)


async def delete_artifact(session: AsyncSession, artifact_id: str) -> None:
    """删 0 行视为 not found（由调用方决定状态码）。"""
    deleted = (
        await session.execute(sa_delete(Artifact).where(Artifact.id == artifact_id))
    ).rowcount
    await session.commit()
    if deleted == 0:
        raise NotFoundError(f"Artifact not found: {artifact_id}")


async def list_artifact_versions(session: AsyncSession, artifact_id: str) -> list[Artifact] | None:
    """版本链：先爬到最远祖先 root，再从 root BFS 收集全部后代，按 version 升序。

    两套 visited 集合独立（「爬上去」的和「长出来」的不能混用），
    parentArtifactId 成环时靠 climbed 防死循环。
    """
    root = await session.get(Artifact, artifact_id)
    if root is None:
        return None

    climbed = {artifact_id}
    while root.parent_artifact_id and root.parent_artifact_id not in climbed:
        parent_id = root.parent_artifact_id
        climbed.add(parent_id)
        parent = await session.get(Artifact, parent_id)
        if parent is None:
            break
        root = parent

    collected = [root]
    visited = {root.id}
    queue = [root.id]
    while queue:
        parent_id = queue.pop(0)
        children = (
            await session.execute(
                select(Artifact).where(Artifact.parent_artifact_id == parent_id)
            )
        ).scalars().all()
        for child in children:
            if child.id in visited:
                continue
            visited.add(child.id)
            collected.append(child)
            queue.append(child.id)

    return sorted(collected, key=lambda row: row.version)


async def create_artifact_version(
    session: AsyncSession, parent_artifact_id: str, raw_content: object, title: str | None
) -> Artifact:
    """以 parent 为父创建 version+1 的新产物行；不存在 → NotFoundError，内容非法 → ServiceError(400)。"""
    parent = await session.get(Artifact, parent_artifact_id)
    if parent is None:
        raise NotFoundError(f"Artifact not found: {parent_artifact_id}")

    content = build_artifact_content(parent.type, raw_content)
    if content is None:
        raise ServiceError(
            describe_artifact_content_error(parent.type, raw_content)
            or f"Invalid content for type {parent.type}"
        )

    artifact = Artifact(
        id=new_artifact_id(),
        conversation_id=parent.conversation_id,
        type=parent.type,
        title=(title or "").strip() or parent.title,
        content=content,
        version=parent.version + 1,
        parent_artifact_id=parent.id,
        created_by_agent_id=parent.created_by_agent_id,
        created_at=now_ms(),
    )
    session.add(artifact)
    await session.commit()
    return artifact
