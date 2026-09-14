"""`/api/platform` —— 纯 UI 提示用，不放敏感信息。"""

from __future__ import annotations

import os

from fastapi import APIRouter

from app.schemas.entities import PlatformResponse

router = APIRouter(prefix="/api/platform", tags=["platform"])


@router.get("", response_model=PlatformResponse)
async def get_platform() -> dict:
    return {"platform": "windows" if os.name == "nt" else "posix"}
