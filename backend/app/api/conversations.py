"""`/api/conversations` —— 会话的列表 / 创建 / 变更 / 删除。

PATCH /{id} 是本阶段最容易写错的点：前端用同一个端点做 5 件事，按 body 字段区分。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.helpers import read_json
from app.db.session import get_session
from app.errors import HttpError, ServiceError
from app.schemas.base import validate_body
from app.schemas.entities import (
    CompactConversationResponse,
    ConversationResponse,
    ConversationsResponse,
    CreateConversationBody,
    DeployConversationBody,
    PatchConversationBody,
)
from app.services import conversation_service, deploy_command_service
from app.services.context_compaction import compact_conversation

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=ConversationsResponse)
async def list_conversations(session: AsyncSession = Depends(get_session)) -> dict:
    return {"conversations": await conversation_service.list_conversations(session)}


@router.post("", status_code=201, response_model=ConversationResponse)
async def create_conversation(request: Request, session: AsyncSession = Depends(get_session)) -> dict:
    body = validate_body(CreateConversationBody, await read_json(request))
    try:
        conversation = await conversation_service.create_conversation(
            session, body.model_dump()
        )
    except ServiceError as err:
        raise HttpError(400, err.message) from err
    return {"conversation": conversation}


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def patch_conversation(
    conversation_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    body = validate_body(PatchConversationBody, await read_json(request))

    # 按固定顺序依次应用，后应用的覆盖前者的返回值 —— 多字段同时出现时就是这个顺序
    conversation = None
    try:
        if body.title is not None:
            conversation = await conversation_service.rename_conversation(
                session, conversation_id, body.title
            )
        if body.add_agent_ids is not None:
            conversation = await conversation_service.add_agents_to_conversation(
                session, conversation_id, body.add_agent_ids
            )
        if body.fs_write_approval_mode is not None:
            conversation = await conversation_service.set_conversation_approval_mode(
                session, conversation_id, body.fs_write_approval_mode
            )
        if body.toggle_pin:
            conversation = await conversation_service.toggle_pin_conversation(
                session, conversation_id
            )
        if body.toggle_archive:
            conversation = await conversation_service.toggle_archive_conversation(
                session, conversation_id
            )
    except ServiceError as err:
        raise HttpError(400, err.message) from err

    assert conversation is not None  # validate_body 已保证至少一个字段存在
    return {"conversation": conversation}


@router.get("/{conversation_id}/deploy")
async def list_deploy_candidates(
    conversation_id: str, session: AsyncSession = Depends(get_session)
) -> dict:
    """可部署的 web_app 产物候选（按创建时间倒序）。"""
    return {
        "candidates": await deploy_command_service.list_deploy_candidates(
            session, conversation_id
        )
    }


@router.post("/{conversation_id}/deploy")
async def deploy_conversation(
    conversation_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    """带 artifactId 部署指定产物；不带则走「唯一候选自动部署 / 多候选返回列表」的判定。"""
    raw = await read_json(request)
    parsed = validate_body(DeployConversationBody, raw if isinstance(raw, dict) else {})
    try:
        return await deploy_command_service.handle_deploy_command(
            session, conversation_id, parsed.artifact_id
        )
    except ServiceError as err:
        raise HttpError(400, err.message) from err


@router.post("/{conversation_id}/compact", response_model=CompactConversationResponse)
async def compact(
    conversation_id: str, session: AsyncSession = Depends(get_session)
) -> dict:
    """压缩较早的会话历史为摘要（错误一律 400，body 形状 { error: message }）。"""
    try:
        return await compact_conversation(session, conversation_id)
    except ServiceError as err:
        raise HttpError(400, err.message) from err


@router.post("/{conversation_id}/regenerate")
async def regenerate_conversation(
    conversation_id: str, session: AsyncSession = Depends(get_session)
) -> dict:
    """删掉最后一条 user 之后的回复，用同一条 user 消息重新触发 agent（错误一律 400）。"""
    try:
        return await conversation_service.regenerate_latest_response(session, conversation_id)
    except ServiceError as err:
        raise HttpError(400, err.message) from err


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str, session: AsyncSession = Depends(get_session)
) -> dict:
    try:
        await conversation_service.delete_conversation(session, conversation_id)
    except ServiceError as err:
        # DELETE 出错一律返回 404（连 not found 也是）
        raise HttpError(404, err.message) from err
    return {"ok": True}
