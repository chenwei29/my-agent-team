"""待审分派计划 store 的行为测试。

契约：批准前用登记的 validator 重编译+校验（依赖从 inputs 推断出来），
校验失败保持 pending 不动；reject/revise 摘表项并把决定 resolve 给等待处。
"""

from __future__ import annotations

from app.schemas.dispatch import DispatchPlanItem
from app.services.dispatch_plan import compile_dispatch_plan, validate_dispatch_plan
from app.services.pending_dispatch_plans import PlanReviewOutcome, pending_dispatch_plans

agents = [{"id": "ag_pm"}, {"id": "ag_frontend"}]


def validate(plan: list[DispatchPlanItem]) -> list[DispatchPlanItem]:
    compiled = compile_dispatch_plan(plan)[0]
    validate_dispatch_plan(compiled, agents, "ag_orchestrator")
    return compiled


class TestPendingDispatchPlansStore:
    def test_approves_the_registered_plan_revalidated_compiled_without_a_body(self):
        pending = pending_dispatch_plans.register(
            conversation_id="conv_plan_review_approve",
            agent_id="ag_orchestrator",
            run_id="run_plan_review_approve",
            plan=[
                DispatchPlanItem(
                    id="t1",
                    agentId="ag_pm",
                    task="Write PRD",
                    expectedOutputs=[{"id": "prd", "type": "document"}],
                ),
                DispatchPlanItem(
                    id="t2",
                    agentId="ag_frontend",
                    task="Build UI",
                    inputs=[{"fromTaskId": "t1", "outputId": "prd"}],
                ),
            ],
            validator=validate,
        )
        resolved: list[PlanReviewOutcome] = []
        pending_dispatch_plans.attach_resolver(pending.id, resolved.append)

        result = pending_dispatch_plans.approve(pending.id)

        assert result == {"ok": True}
        assert resolved and resolved[0]["kind"] == "approve"
        if resolved and resolved[0]["kind"] == "approve":
            # compile_dispatch_plan 从 inputs 推断出 dependsOn
            assert resolved[0]["plan"][1].dependsOn == ["t1"]
        assert pending_dispatch_plans.get(pending.id) is None

    def test_keeps_invalid_plans_pending_on_approve(self):
        pending = pending_dispatch_plans.register(
            conversation_id="conv_plan_review_invalid",
            agent_id="ag_orchestrator",
            run_id="run_plan_review_invalid",
            plan=[DispatchPlanItem(id="t1", agentId="ag_missing", task="Write PRD")],
            validator=validate,
        )
        resolved: list[PlanReviewOutcome] = []
        pending_dispatch_plans.attach_resolver(pending.id, resolved.append)

        result = pending_dispatch_plans.approve(pending.id)

        assert result["ok"] is False
        assert resolved == []
        assert pending_dispatch_plans.get(pending.id) is not None
        assert pending_dispatch_plans.reject(pending.id) is True

    def test_resolves_rejection_and_removes_the_pending_plan(self):
        pending = pending_dispatch_plans.register(
            conversation_id="conv_plan_review_reject",
            agent_id="ag_orchestrator",
            run_id="run_plan_review_reject",
            plan=[DispatchPlanItem(id="t1", agentId="ag_pm", task="Write PRD")],
            validator=validate,
        )
        resolved: list[PlanReviewOutcome] = []
        pending_dispatch_plans.attach_resolver(pending.id, resolved.append)

        assert pending_dispatch_plans.reject(pending.id) is True
        assert resolved and resolved[0]["kind"] == "reject"
        assert pending_dispatch_plans.get(pending.id) is None

    def test_resolves_revise_with_the_feedback_and_removes_the_pending_plan(self):
        pending = pending_dispatch_plans.register(
            conversation_id="conv_plan_review_revise",
            agent_id="ag_orchestrator",
            run_id="run_plan_review_revise",
            plan=[DispatchPlanItem(id="t1", agentId="ag_pm", task="Write PRD")],
            validator=validate,
        )
        resolved: list[PlanReviewOutcome] = []
        pending_dispatch_plans.attach_resolver(pending.id, resolved.append)

        assert pending_dispatch_plans.revise(pending.id, "t2 依赖 t1") is True
        assert resolved == [{"kind": "revise", "feedback": "t2 依赖 t1"}]
        assert pending_dispatch_plans.get(pending.id) is None
