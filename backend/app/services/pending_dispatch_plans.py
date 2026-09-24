"""待审分派计划的中转站：注册 → 等用户审批 → resolve 回等待中的编排 run。

生命周期与其它审批流一致（内存注册表，重启即丢）：

- ``register`` 落表项并广播 ``dispatch.plan.pending``，前端弹出计划审批卡片；
- 用户的决定（approve / reject / revise）经 HTTP 端点打进来，``_finalize`` 摘表项、
  广播 ``dispatch.plan.resolved``、把结果 resolve 给 Orchestrator run 的等待处；
- 批准会再过一遍登记的 validator（编译 + 语义校验）做防御 —— 校验失败时
  **保持 pending 不动**（用户可改可拒），错误原样回给端点；
- 修改（revise）把用户的自然语言反馈交回编排 run 重排，当前 pending 作废
  （重排产出的新计划会再次走 register）。

``_finalize`` 是 resolver 的唯一写入方，先摘表项再回调，谁先到谁生效。
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from app.schemas.dispatch import DispatchPlanItem, PendingDispatchPlan
from app.schemas.events import DispatchPlanPendingEvent, DispatchPlanResolvedEvent
from app.services.event_bus import event_bus
from app.utils.ids import new_pending_dispatch_plan_id
from app.utils.time import now_ms

PlanValidator = Callable[[list[DispatchPlanItem]], list[DispatchPlanItem]]


class PlanReviewOutcome(TypedDict, total=False):
    """用户对 pending 计划的决定，由等待处回传给编排 run。"""

    kind: str  # 'approve' | 'reject' | 'revise'
    plan: list[DispatchPlanItem]
    feedback: str


class PendingDispatchPlanResult(TypedDict, total=False):
    ok: bool
    error: str


class _Entry:
    __slots__ = ("pending_plan", "resolver", "validator")

    def __init__(self, pending_plan: PendingDispatchPlan, validator: PlanValidator) -> None:
        self.pending_plan = pending_plan
        self.resolver: Callable[[PlanReviewOutcome], Any] | None = None
        self.validator = validator


class PendingDispatchPlansStore:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(
        self,
        *,
        conversation_id: str,
        agent_id: str,
        run_id: str,
        plan: list[DispatchPlanItem],
        validator: PlanValidator,
    ) -> PendingDispatchPlan:
        pending_plan = PendingDispatchPlan(
            id=new_pending_dispatch_plan_id(),
            conversationId=conversation_id,
            agentId=agent_id,
            runId=run_id,
            plan=plan,
            createdAt=now_ms(),
        )
        self._entries[pending_plan.id] = _Entry(pending_plan, validator)
        event_bus.publish(
            DispatchPlanPendingEvent(
                conversationId=conversation_id,
                timestamp=pending_plan.createdAt,
                pendingPlan=pending_plan,
            )
        )
        return pending_plan

    def attach_resolver(self, pending_id: str, resolver: Callable[[PlanReviewOutcome], Any]) -> bool:
        entry = self._entries.get(pending_id)
        if entry is None:
            return False
        entry.resolver = resolver
        return True

    def get(self, pending_id: str) -> PendingDispatchPlan | None:
        entry = self._entries.get(pending_id)
        return entry.pending_plan if entry else None

    def list_by_conversation(self, conversation_id: str) -> list[PendingDispatchPlan]:
        items = [
            entry.pending_plan
            for entry in self._entries.values()
            if entry.pending_plan.conversationId == conversation_id
        ]
        items.sort(key=lambda p: p.createdAt)
        return items

    def approve(self, pending_id: str) -> PendingDispatchPlanResult:
        """批准：用已登记的（只读）计划执行；仍过一遍 validator 做防御性校验。"""
        entry = self._entries.get(pending_id)
        if entry is None:
            return {"ok": False, "error": "Pending dispatch plan not found"}

        try:
            compiled_plan = entry.validator(entry.pending_plan.plan)
        except Exception as err:  # 校验失败保持 pending，用户可改可拒
            return {"ok": False, "error": str(err)}

        if entry.resolver is not None:
            entry.resolver({"kind": "approve", "plan": compiled_plan})
        self._finalize(pending_id, approved=True)
        return {"ok": True}

    def revise(self, pending_id: str, feedback: str) -> bool:
        """修改：反馈交回编排 run 重排；当前 pending 作废（重排后会再发新的）。"""
        entry = self._entries.get(pending_id)
        if entry is None:
            return False
        if entry.resolver is not None:
            entry.resolver({"kind": "revise", "feedback": feedback})
        self._finalize(pending_id, approved=False, revising=True)
        return True

    def reject(self, pending_id: str) -> bool:
        entry = self._entries.get(pending_id)
        if entry is None:
            return False
        if entry.resolver is not None:
            entry.resolver({"kind": "reject"})
        self._finalize(pending_id, approved=False)
        return True

    def cancel(self, pending_id: str) -> None:
        """abort 路径：按拒绝收口（会发 resolved 事件，前端随之关卡片）。"""
        entry = self._entries.get(pending_id)
        if entry is None:
            return
        if entry.resolver is not None:
            entry.resolver({"kind": "reject"})
        self._finalize(pending_id, approved=False)

    def _finalize(self, pending_id: str, *, approved: bool, revising: bool = False) -> None:
        entry = self._entries.pop(pending_id, None)
        if entry is None:
            return
        event_bus.publish(
            DispatchPlanResolvedEvent(
                conversationId=entry.pending_plan.conversationId,
                timestamp=now_ms(),
                pendingId=pending_id,
                runId=entry.pending_plan.runId,
                approved=approved,
                revising=True if revising else None,
            )
        )


pending_dispatch_plans = PendingDispatchPlansStore()
