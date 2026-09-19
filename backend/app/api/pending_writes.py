"""fs_write 审批端点。

GET 用于页面刷新后恢复面板；POST 由用户「批准/拒绝」按钮触发。
按既有契约：这条路由不校验 pwId 与路径里 conversationId 的从属关系
（写操作本身已被 workspace 行限定），未知 id 一律 404。
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import ValidationError

from app.errors import HttpError, InvalidBody
from app.schemas.entities import ResolvePendingBody
from app.services.pending_writes import pending_writes

router = APIRouter(prefix="/api/conversations/{conversation_id}/pending-writes", tags=["pending-writes"])


@router.get("")
async def list_pending_writes(conversation_id: str) -> dict:
    return {"pendingWrites": pending_writes.list_by_conversation(conversation_id)}


@router.post("/{pending_id}")
async def resolve_pending_write(conversation_id: str, pending_id: str, body: dict) -> dict:
    try:
        parsed = ResolvePendingBody.model_validate(body)
    except ValidationError as err:
        issues = [{"path": list(e.get("loc", [])), "message": e.get("msg", "")} for e in err.errors()]
        raise InvalidBody(issues) from err

    if pending_writes.get(pending_id) is None:
        raise HttpError(404, "Pending write not found")

    if parsed.action == "approve":
        ok = pending_writes.approve(pending_id)
    else:
        ok = pending_writes.reject(pending_id)

    if not ok:
        # approve 时落盘失败（store 已按拒绝收口，工具侧看到拒绝文案）
        raise HttpError(500, "Failed to process pending write")
    return {"ok": True}
