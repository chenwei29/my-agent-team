"""`GET /api/search` —— 消息全文搜索。

⚠️ 这个端点的响应信封与全项目其他端点不同：
- 成功：`{ ok: true, data: { hits, total, tookMs } }`
- 失败：`{ ok: false, error: { code, message } }` + 400
前端 `searchMessagesApi` 直接解构 `{ ok, data }`，改形状就白屏。
查询参数校验也不走全局的 InvalidBody 形状，错误码统一 `INVALID_QUERY`。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.services.search_service import search_messages

router = APIRouter(prefix="/api/search", tags=["search"])

# 与 web/src 的调用口径一致：q 1~200 字符，limit 1~100，offset 0+
_Q_MIN, _Q_MAX = 1, 200
_LIMIT_DEFAULT, _LIMIT_MIN, _LIMIT_MAX = 20, 1, 100


def _err(code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400, content={"ok": False, "error": {"code": code, "message": message}}
    )


@router.get("")
async def search(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> Any:
    params = request.query_params

    q = params.get("q")
    if q is None or not (_Q_MIN <= len(q) <= _Q_MAX):
        return _err("INVALID_QUERY", "q must be 1-200 characters")

    limit_value = _LIMIT_DEFAULT
    if (raw_limit := params.get("limit")) is not None:
        try:
            limit_value = int(raw_limit)
        except ValueError:
            return _err("INVALID_QUERY", "limit must be an integer")
        if not (_LIMIT_MIN <= limit_value <= _LIMIT_MAX):
            return _err("INVALID_QUERY", "limit must be 1-100")

    offset_value = 0
    if (raw_offset := params.get("offset")) is not None:
        try:
            offset_value = int(raw_offset)
        except ValueError:
            return _err("INVALID_QUERY", "offset must be an integer")
        if offset_value < 0:
            return _err("INVALID_QUERY", "offset must be >= 0")

    role = params.get("role")
    if role is not None and role not in ("user", "agent"):
        return _err("INVALID_QUERY", "role must be 'user' or 'agent'")

    fallback = params.get("fallback")
    if fallback is not None and fallback != "like":
        return _err("INVALID_QUERY", "fallback must be 'like'")

    result = await search_messages(
        session,
        query=q,
        limit=limit_value,
        offset=offset_value,
        conversation_id=params.get("conversationId"),
        role=role,
        fallback=fallback,
    )
    if result.get("error") == "INVALID_QUERY":
        return _err("INVALID_QUERY", "Invalid search syntax")

    return {
        "ok": True,
        "data": {
            "hits": result["hits"],
            "total": result["total"],
            "tookMs": result["tookMs"],
        },
    }
