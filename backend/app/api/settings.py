"""`/api/settings` —— 读取与写入全局设置（单例行）。

GET 会原样返回 key 字段（用户已自行选择填明文），不做脱敏。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.helpers import read_json
from app.db.session import get_session
from app.schemas.base import validate_body
from app.schemas.entities import SettingsResponse, UpdateSettingsBody
from app.services import settings_service

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
async def get_settings(session: AsyncSession = Depends(get_session)) -> dict:
    return {"settings": await settings_service.get_app_settings(session)}


@router.patch("", response_model=SettingsResponse)
async def update_settings(request: Request, session: AsyncSession = Depends(get_session)) -> dict:
    body = validate_body(UpdateSettingsBody, await read_json(request))
    settings = await settings_service.update_app_settings(
        session, body.model_dump(), set(body.model_fields_set)
    )
    return {"settings": settings}
