"""`/api/agents` —— 自定义 Agent 的列表 / 创建 / 改名 / 删除。

⚠️ adapter_name='mock' 无法通过这里创建（CreateAgentBody 的 enum 不含 mock），
E2E 专用 mock agent 只能由 bootstrap_cli 直接写库 —— 请求体校验只接受 enum 内的值。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.helpers import read_json
from app.db.session import get_session
from app.errors import HttpError, ServiceError
from app.schemas.base import validate_body
from app.schemas.entities import (
    AgentResponse,
    AgentsResponse,
    CreateAgentBody,
    UpdateAgentBody,
)
from app.services import agent_service

router = APIRouter(prefix="/api/agents", tags=["agents"])

# 这三个端点出错一律返回 400（连 not found 也是）
_BAD_REQUEST = 400


@router.get("", response_model=AgentsResponse)
async def list_agents(session: AsyncSession = Depends(get_session)) -> dict:
    return {"agents": await agent_service.list_agents_ordered(session)}


@router.post("/draft", status_code=501)
async def create_agent_draft() -> Response:
    """P1 先占位：agent 草稿生成要真调 LLM，属 P3。"""
    return Response(
        content='{"error":"Agent draft generation is not implemented yet (phase 3)"}',
        media_type="application/json",
        status_code=501,
    )


@router.post("", status_code=201, response_model=AgentResponse)
async def create_agent(request: Request, session: AsyncSession = Depends(get_session)) -> dict:
    body = validate_body(CreateAgentBody, await read_json(request))
    try:
        agent = await agent_service.create_custom_agent(session, body.model_dump())
    except ServiceError as err:
        raise HttpError(_BAD_REQUEST, err.message) from err
    return {"agent": agent}


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    body = validate_body(UpdateAgentBody, await read_json(request))
    try:
        agent = await agent_service.update_custom_agent(
            session, agent_id, body.model_dump(), set(body.model_fields_set)
        )
    except ServiceError as err:
        raise HttpError(_BAD_REQUEST, err.message) from err
    return {"agent": agent}


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    try:
        await agent_service.delete_custom_agent(session, agent_id)
    except ServiceError as err:
        raise HttpError(_BAD_REQUEST, err.message) from err
    return {"ok": True}
