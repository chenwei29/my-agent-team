"""`/api/messages/{id}` 上的消息高级操作：withdraw / edit / pin / bookmark。

状态码口径（与前端调用方约定）：
- withdraw / edit：错误消息里含 "not found" → 404，其余 → 400
- pin / bookmark：一律 400（包括 not found）
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.helpers import read_json
from app.db.session import get_session
from app.errors import HttpError, ServiceError
from app.schemas.base import validate_body
from app.schemas.entities import EditMessageBody, WithdrawMessageBody
from app.services import conversation_service

router = APIRouter(prefix="/api/messages", tags=["message-ops"])


def _message_error_status(message: str) -> int:
    return 404 if "not found" in message else 400


@router.post("/{message_id}/withdraw")
async def withdraw_message(
    message_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    body = validate_body(WithdrawMessageBody, await read_json(request))
    try:
        return await conversation_service.withdraw_latest_user_message(
            session, body.conversation_id, message_id
        )
    except ServiceError as err:
        raise HttpError(_message_error_status(err.message), err.message) from err


@router.post("/{message_id}/edit")
async def edit_message(
    message_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    body = validate_body(EditMessageBody, await read_json(request))
    try:
        return await conversation_service.edit_and_resend_latest_user_message(
            session, body.conversation_id, message_id, body.content
        )
    except ServiceError as err:
        raise HttpError(_message_error_status(err.message), err.message) from err


@router.post("/{message_id}/pin")
async def pin_message(
    message_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    body = validate_body(WithdrawMessageBody, await read_json(request))
    try:
        return await conversation_service.toggle_pinned_message(
            session, body.conversation_id, message_id
        )
    except ServiceError as err:
        raise HttpError(400, err.message) from err


@router.post("/{message_id}/bookmark")
async def bookmark_message(
    message_id: str, request: Request, session: AsyncSession = Depends(get_session)
) -> dict:
    body = validate_body(WithdrawMessageBody, await read_json(request))
    try:
        return await conversation_service.toggle_bookmarked_message(
            session, body.conversation_id, message_id
        )
    except ServiceError as err:
        raise HttpError(400, err.message) from err
