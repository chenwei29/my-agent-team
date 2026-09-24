"""`GET /api/usage/summary` —— 全局 token 用量聚合（设置面板的分析 Tab）。

响应是聚合对象本身（没有 {ok}/{settings} 这类外层信封），字段见 schemas/entities.py。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.schemas.entities import UsageSummaryOut
from app.services import usage_service

router = APIRouter(prefix="/api/usage", tags=["usage"])


@router.get("/summary", response_model=UsageSummaryOut)
async def usage_summary(session: AsyncSession = Depends(get_session)) -> dict:
    return await usage_service.build_usage_summary(session)
