"""待审分派计划的查询与审批端点。

GET 用于页面刷新后恢复计划审批卡片；POST 由用户「批准 / 拒绝 / 修改」按钮触发。
修改（revise）会把反馈落成一条 user 消息（进对话、跨端可见）再交回等待中的
编排 run 重排，不触发新 run。planId 不存在或与会话不匹配一律 404。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.helpers import read_json
from app.db.session import get_session
from app.errors import HttpError
from app.schemas.base import validate_body
from app.schemas.entities import ReviewDispatchPlanBody
from app.services import conversation_service
from app.services.pending_dispatch_plans import pending_dispatch_plans

router = APIRouter(
    prefix="/api/conversations/{conversation_id}/pending-dispatch-plans",
    tags=["pending-dispatch-plans"],
)


@router.get("")
async def list_pending_dispatch_plans(conversation_id: str) -> dict:
    return {
        "pendingDispatchPlans": [
            pending.model_dump() for pending in pending_dispatch_plans.list_by_conversation(conversation_id)
        ]
    }


@router.post("/{plan_id}")
async def resolve_pending_dispatch_plan(
    conversation_id: str,
    plan_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict:
    body = validate_body(ReviewDispatchPlanBody, await read_json(request))

    existing = pending_dispatch_plans.get(plan_id)
    if existing is None or existing.conversationId != conversation_id:
        raise HttpError(404, "Pending dispatch plan not found")

    if body.action == "reject":
        ok = pending_dispatch_plans.reject(plan_id)
        if not ok:
            raise HttpError(500, "Failed to reject pending dispatch plan")
        return {"ok": True}

    if body.action == "revise":
        result = await conversation_service.revise_dispatch_plan(
            session, conversation_id, plan_id, body.feedback
        )
        if not result["ok"]:
            raise HttpError(400, result["error"])
        return {"ok": True}

    result = pending_dispatch_plans.approve(plan_id)
    if not result["ok"]:
        raise HttpError(400, result["error"])
    return {"ok": True}
