"""bash 命令审批端点。

这条路由与 fs_write 审批不同：**校验 commandId 与路径里 conversationId
的从属关系**（不一致按 404 处理）。
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import ValidationError

from app.errors import HttpError, InvalidBody
from app.schemas.entities import ResolvePendingBody
from app.services.pending_bash_commands import pending_bash_commands

router = APIRouter(
    prefix="/api/conversations/{conversation_id}/pending-bash-commands", tags=["pending-bash-commands"]
)


@router.get("")
async def list_pending_commands(conversation_id: str) -> dict:
    return {"pendingCommands": pending_bash_commands.list_by_conversation(conversation_id)}


@router.post("/{pending_id}")
async def resolve_pending_command(conversation_id: str, pending_id: str, body: dict) -> dict:
    try:
        parsed = ResolvePendingBody.model_validate(body)
    except ValidationError as err:
        issues = [{"path": list(e.get("loc", [])), "message": e.get("msg", "")} for e in err.errors()]
        raise InvalidBody(issues) from err

    existing = pending_bash_commands.get(pending_id)
    if existing is None or existing.get("conversationId") != conversation_id:
        raise HttpError(404, "Pending command not found")

    if parsed.action == "approve":
        ok = pending_bash_commands.approve(pending_id)
    else:
        ok = pending_bash_commands.reject(pending_id)

    if not ok:
        raise HttpError(500, "Failed to process pending command")
    return {"ok": True}
