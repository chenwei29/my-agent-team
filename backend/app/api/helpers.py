"""路由层小工具。"""

from __future__ import annotations

from typing import Any

from fastapi import Request


async def read_json(request: Request) -> Any:
    """读请求体 JSON —— 体不是合法 JSON 时返回 None，
    由后续 validate_body 统一报 400 Invalid body（而不是 FastAPI 默认的 422）。"""
    try:
        return await request.json()
    except Exception:  # noqa: BLE001 - 任何解析失败都按「没有 body」处理
        return None
