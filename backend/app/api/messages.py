"""`/api/conversations/{id}/messages` —— 会话消息的列表 / 发送 / 清空历史。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.helpers import read_json
from app.db.session import get_session
from app.errors import ConflictError, HttpError, NotFoundError, ServiceError
from app.schemas.base import validate_body
from app.schemas.entities import (
    ClearHistoryResponse,
    MessagesResponse,
    SendMessageBody,
    SendMessageResponse,
)
from app.services import conversation_service

router = APIRouter(prefix="/api/conversations/{conversation_id}/messages", tags=["messages"])


@router.get("", response_model=MessagesResponse)
async def list_messages(
    conversation_id: str, session: AsyncSession = Depends(get_session)
) -> dict:
    # 这个端点不捕获异常：会话不存在也返回 200 + 空数组
    return {"messages": await conversation_service.list_messages(session, conversation_id)}


# response_model_exclude_none：P1 不产 run，messages 字段没有值，不应该以 "messages": null 的
# 形式出现在响应里 —— 前端 api.ts 的类型是 `messages?: MessageRow[]`，多一个显式 null 是白送的
# 契约偏差。P2 真正带上 messages 时这个开关仍然成立，所以不用改回来。
@router.post("", status_code=202, response_model=SendMessageResponse, response_model_exclude_none=True)
async def send_message(
    conversation_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    body = validate_body(SendMessageBody, await read_json(request))
    try:
        result = await conversation_service.send_message(
            session, {"conversation_id": conversation_id, **body.model_dump()}
        )
    except ServiceError as err:
        raise HttpError(400, err.message) from err
    return result


@router.delete("", response_model=ClearHistoryResponse)
async def clear_history(
    conversation_id: str, session: AsyncSession = Depends(get_session)
) -> dict:
    try:
        return await conversation_service.clear_conversation_history(session, conversation_id)
    except NotFoundError as err:
        raise HttpError(404, err.message) from err
    except ConflictError as err:
        raise HttpError(409, err.message) from err
    except ServiceError as err:
        raise HttpError(400, err.message) from err
