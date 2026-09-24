"""「部署」这条对话指令的判定与编排。

用户在输入框里发 `/deploy` / `部署` / `发布` / `上线`（可带 `art_xxx`）时，不触发任何 agent run，
而是当场把候选产物部署掉，并往会话里插一条 system 消息（带 deploy_status / deploy_candidates part）
让前端渲染部署卡片。

返回三种形态（前端按 kind 分支）：no_candidates / candidate_selection / deployed。
没有 web_app 产物时还会去工作区找 dist / build / out 这类现成输出目录 —— 本地跑过构建的项目
可以直接部署，不用先让 agent 重新产一遍。
"""

from __future__ import annotations

import os
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Artifact, Conversation, Message, Workspace
from app.errors import ServiceError
from app.schemas.events import MessageAddedEvent, MessageRecord
from app.security.workspace_utils import get_effective_cwd
from app.services.event_bus import event_bus
from app.tools.deploy import (
    DeployWorkspaceArgs,
    deploy_artifact_for_conversation,
    deploy_workspace_for_conversation,
)
from app.utils.ids import new_message_id
from app.utils.time import now_ms

DEPLOY_COMMAND_RE = re.compile(r"^(?:/deploy|部署|发布|上线)(?:\s+(art_[0-9A-Za-z]+))?$", re.IGNORECASE)

WORKSPACE_DEPLOY_CANDIDATES = (
    "dist",
    "build",
    "out",
    "public",
    "client/dist",
    "client/build",
    "client/out",
    "apps/web/dist",
    "apps/web/build",
    "apps/web/out",
)

NO_CANDIDATE_TEXT = (
    "当前会话还没有可部署的网页产物，也没有找到常见的本地静态输出目录"
    "（dist/build/out/client/dist）。请先让 Agent 生成 web_app 产物，或构建本地项目后再部署。"
)


def parse_deploy_command(content: str) -> dict[str, str] | None:
    """识别部署指令；不是指令返回 None（空 dict 表示「指令但没指定产物」）。"""
    match = DEPLOY_COMMAND_RE.match(content.strip())
    if not match:
        return None
    return {"artifact_id": match.group(1)} if match.group(1) else {}


def decide_deploy_command(
    candidates: list[dict[str, Any]], artifact_id: str | None
) -> dict[str, Any]:
    if artifact_id:
        return {"kind": "deploy", "artifact_id": artifact_id}
    if not candidates:
        return {"kind": "no_candidates"}
    if len(candidates) == 1:
        return {"kind": "deploy", "artifact_id": candidates[0]["artifactId"]}
    return {"kind": "select", "candidates": candidates}


async def list_deploy_candidates(
    session: AsyncSession, conversation_id: str
) -> list[dict[str, Any]]:
    rows = list(
        await session.scalars(
            select(Artifact)
            .where(Artifact.conversation_id == conversation_id, Artifact.type == "web_app")
            .order_by(Artifact.created_at.desc())
        )
    )
    return [
        {
            "artifactId": row.id,
            "title": row.title,
            "version": row.version,
            "createdByAgentId": row.created_by_agent_id,
            "createdAt": row.created_at,
        }
        for row in rows
    ]


async def handle_deploy_command(
    session: AsyncSession,
    conversation_id: str,
    artifact_id: str | None = None,
    after_created_at: int | None = None,
) -> dict[str, Any]:
    candidates = [] if artifact_id else await list_deploy_candidates(session, conversation_id)
    decision = decide_deploy_command(candidates, artifact_id)

    if decision["kind"] == "no_candidates":
        workspace_deploy = await _deploy_first_workspace_candidate(
            session, conversation_id, after_created_at
        )
        if workspace_deploy is not None:
            return workspace_deploy

        message = await _insert_system_message(
            session,
            conversation_id,
            [{"type": "text", "content": NO_CANDIDATE_TEXT}],
            after_created_at,
        )
        return {"kind": "no_candidates", "candidates": [], "message": message}

    if decision["kind"] == "select":
        message = await _insert_system_message(
            session,
            conversation_id,
            [{"type": "deploy_candidates", "candidates": decision["candidates"]}],
            after_created_at,
        )
        return {
            "kind": "candidate_selection",
            "candidates": decision["candidates"],
            "message": message,
        }

    return await deploy_selected_artifact(
        session, conversation_id, decision["artifact_id"], after_created_at
    )


async def deploy_selected_artifact(
    session: AsyncSession,
    conversation_id: str,
    artifact_id: str,
    after_created_at: int | None = None,
) -> dict[str, Any]:
    deployment = await deploy_artifact_for_conversation(session, conversation_id, artifact_id)
    message = await _insert_system_message(
        session,
        conversation_id,
        [{"type": "deploy_status", "deployment": deployment}],
        after_created_at,
    )
    return {"kind": "deployed", "deployment": deployment, "message": message}


async def _deploy_first_workspace_candidate(
    session: AsyncSession, conversation_id: str, after_created_at: int | None
) -> dict[str, Any] | None:
    candidate = await _find_workspace_deploy_candidate(session, conversation_id)
    if candidate is None:
        return None

    deployment = await deploy_workspace_for_conversation(
        session,
        conversation_id,
        DeployWorkspaceArgs(path=candidate["path"], title=candidate["title"]),
    )
    message = await _insert_system_message(
        session,
        conversation_id,
        [{"type": "deploy_status", "deployment": deployment}],
        after_created_at,
    )
    return {"kind": "deployed", "deployment": deployment, "message": message}


async def _find_workspace_deploy_candidate(
    session: AsyncSession, conversation_id: str
) -> dict[str, str] | None:
    """按固定顺序找第一个「存在且带 index.html」的静态输出目录。"""
    workspace = await session.scalar(
        select(Workspace).where(Workspace.conversation_id == conversation_id)
    )
    if workspace is None:
        return None

    cwd = get_effective_cwd(workspace)
    for rel_path in WORKSPACE_DEPLOY_CANDIDATES:
        abs_path = os.path.join(cwd, rel_path)
        if not os.path.isdir(abs_path):
            continue
        if not os.path.isfile(os.path.join(abs_path, "index.html")):
            continue
        return {"path": rel_path, "title": f"Workspace {rel_path}"}
    return None


async def _insert_system_message(
    session: AsyncSession,
    conversation_id: str,
    parts: list[dict[str, Any]],
    after_created_at: int | None,
) -> dict[str, Any]:
    """插一条 system 消息并广播。

    createdAt 取 `max(now, 触发消息的 createdAt + 1)`：同毫秒内落库会让「按时间排序」
    把这条系统消息排到触发它的用户消息前面。
    """
    conversation = await session.scalar(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    if conversation is None:
        raise ServiceError(f"Conversation not found: {conversation_id}")

    created_at = max(now_ms(), (after_created_at or 0) + 1)
    row = Message(
        id=new_message_id(),
        conversation_id=conversation_id,
        role="system",
        agent_id=None,
        parts=parts,
        status="complete",
        parent_message_id=None,
        mentioned_agent_ids=[],
        run_id=None,
        usage=None,
        created_at=created_at,
    )
    session.add(row)
    conversation.updated_at = created_at
    await session.commit()

    record = {
        "id": row.id,
        "conversationId": conversation_id,
        "role": "system",
        "agentId": None,
        "parts": parts,
        "status": "complete",
        "parentMessageId": None,
        "mentionedAgentIds": [],
        "runId": None,
        "usage": None,
        "createdAt": created_at,
    }
    event_bus.publish(
        MessageAddedEvent(
            conversationId=conversation_id,
            timestamp=created_at,
            message=MessageRecord(**record),
        )
    )
    return record
