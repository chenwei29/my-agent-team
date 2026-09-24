"""编排全流程集成测试：mock adapter 驱动 PLAN → 审批 → DAG 波次 → AGGREGATE。

不起真实 LLM：编排 Agent 与子 Agent 都是 mock adapter，靠工具集 / 阶段标记选脚本
（plan_tasks → 吐计划；report_task_result → 上报完成；聚合阶段 → 总结发言）。
审批由后台 watcher 自动批准 / 提交修改意见，模拟前端计划卡片的用户决定。
"""

from __future__ import annotations

import asyncio
import time

from httpx import AsyncClient
from sqlalchemy import select

from app.db.models import Agent, AgentRun
from app.db.session import SessionLocal
from app.services.pending_dispatch_plans import pending_dispatch_plans
from app.utils.time import now_ms
from tests.helpers import EventCollector, messages_of, wait_for_conversation_runs

ORCHESTRATOR_ID = "ag_orch_mock"
WORKER_IDS = ["ag_worker_1", "ag_worker_2", "ag_worker_3"]


async def _ensure_agents() -> None:
    """编排者 + 三个 mock worker（计划脚本固定派给这三个 id）。"""
    specs = [
        {
            "id": ORCHESTRATOR_ID,
            "name": "编排 Mock",
            "avatar": "🎛️",
            "description": "编排集成测试用",
            "capabilities": ["orchestration"],
            "system_prompt": "你是编排者，负责拆解并分派任务。",
            "tool_names": [
                "plan_tasks",
                "ask_user",
                "fs_list",
                "fs_read",
                "read_attachment",
                "read_artifact",
            ],
            "is_orchestrator": True,
        },
        *[
            {
                "id": worker_id,
                "name": f"工人 {index + 1}",
                "avatar": "🧰",
                "description": "子任务执行者",
                "capabilities": ["docs"],
                "system_prompt": "你是子任务执行者。",
                "tool_names": [],
                "is_orchestrator": False,
            }
            for index, worker_id in enumerate(WORKER_IDS)
        ],
    ]
    async with SessionLocal() as session:
        for spec in specs:
            existing = await session.scalar(select(Agent).where(Agent.id == spec["id"]))
            if existing is not None:
                continue
            session.add(
                Agent(
                    id=spec["id"],
                    name=spec["name"],
                    avatar=spec["avatar"],
                    description=spec["description"],
                    capabilities=spec["capabilities"],
                    system_prompt=spec["system_prompt"],
                    adapter_name="mock",
                    model_provider=None,
                    model_id=None,
                    api_key=None,
                    api_base_url=None,
                    tool_names=spec["tool_names"],
                    is_builtin=False,
                    is_orchestrator=spec["is_orchestrator"],
                    supports_vision=False,
                    created_at=now_ms(),
                )
            )
        await session.commit()


async def _create_group(client: AsyncClient) -> dict:
    res = await client.post(
        "/api/conversations",
        json={"mode": "group", "agentIds": [ORCHESTRATOR_ID, *WORKER_IDS]},
    )
    assert res.status_code == 201, res.text
    return res.json()["conversation"]


async def _drain_pending(conversation_id: str) -> None:
    """清掉残留待审项，避免上个用例失败把状态带给下一个。"""
    for pending in pending_dispatch_plans.list_by_conversation(conversation_id):
        pending_dispatch_plans.cancel(pending.id)


async def _watch_reviews(
    conversation_id: str,
    decisions: list[tuple[str, str | None]],
    *,
    timeout: float = 20.0,
) -> None:
    """按序处理待审计划：('approve', None) 或 ('revise', feedback)。"""
    deadline = time.monotonic() + timeout
    handled = 0
    while time.monotonic() < deadline and handled < len(decisions):
        for pending in pending_dispatch_plans.list_by_conversation(conversation_id):
            decision, feedback = decisions[handled]
            if decision == "approve":
                result = pending_dispatch_plans.approve(pending.id)
                assert result.get("ok"), result
            else:
                assert pending_dispatch_plans.revise(pending.id, feedback or "")
            handled += 1
            break
        await asyncio.sleep(0.05)
    assert handled == len(decisions), (
        f"只处理了 {handled}/{len(decisions)} 份待审计划"
    )


def _event_index(events: list, type_name: str, matcher) -> int:
    for index, event in enumerate(events):
        if event.type == type_name and matcher(event):
            return index
    raise AssertionError(f"找不到事件 {type_name}")


def _agent_text(messages, agent_id: str, needle: str) -> str:
    for message in reversed(messages):
        if message.agent_id != agent_id:
            continue
        for part in message.parts or []:
            if isinstance(part, dict) and part.get("type") == "text":
                if needle in part.get("content", ""):
                    return part["content"]
    raise AssertionError(f"{agent_id} 的消息里找不到「{needle}」")


async def test_full_orchestration_runs_waves_and_aggregates(client: AsyncClient):
    await _ensure_agents()
    conv = await _create_group(client)
    await _drain_pending(conv["id"])

    async with EventCollector() as collector:
        watcher = asyncio.create_task(_watch_reviews(conv["id"], [("approve", None)]))
        res = await client.post(
            f"/api/conversations/{conv['id']}/messages",
            json={"content": "整理一份要点文档"},
        )
        assert res.status_code == 202, res.text
        await wait_for_conversation_runs(conv["id"], min_runs=4)
        await watcher

    events = collector.events

    # 计划卡片：任务 / 依赖 / 执行者齐了
    pendings = collector.of_type("dispatch.plan.pending")
    assert len(pendings) == 1
    plan = pendings[0].pendingPlan.plan
    assert [t.id for t in plan] == ["t1", "t2", "t3"]
    assert [t.agentId for t in plan] == WORKER_IDS
    assert plan[0].dependsOn in (None, [])
    assert plan[1].dependsOn == ["t1"] and plan[2].dependsOn == ["t1"]

    resolved = collector.of_type("dispatch.plan.resolved")
    assert len(resolved) == 1 and resolved[0].approved is True

    # 批准后广播 dispatch.plan
    plans = collector.of_type("dispatch.plan")
    assert len(plans) == 1 and [t.id for t in plans[0].plan] == ["t1", "t2", "t3"]

    # 波次：t1 先跑完，t2/t3 并发起跑（start 交错在对方 end 之前）
    events = collector.events
    start = lambda task_id: _event_index(  # noqa: E731
        events, "dispatch.start", lambda e: e.taskId == task_id
    )
    end = lambda task_id: _event_index(  # noqa: E731
        events, "dispatch.end", lambda e: e.taskId == task_id
    )
    assert end("t1") < start("t2") and end("t1") < start("t3")
    assert start("t2") < end("t3") and start("t3") < end("t2")

    ends = collector.of_type("dispatch.end")
    assert len(ends) == 3
    assert {e.taskId: e.status for e in ends} == {
        "t1": "complete",
        "t2": "complete",
        "t3": "complete",
    }
    assert all(e.parentRunId for e in collector.of_type("dispatch.start"))

    # 全部 complete 不进补救轮，run 收完整态
    async with SessionLocal() as session:
        rows = list(
            await session.scalars(
                select(AgentRun).where(AgentRun.conversation_id == conv["id"])
            )
        )
    parent_rows = [r for r in rows if r.parent_run_id is None]
    child_rows = [r for r in rows if r.parent_run_id is not None]
    assert len(parent_rows) == 1 and parent_rows[0].status == "complete"
    assert len(child_rows) == 3 and all(r.status == "complete" for r in child_rows)

    # 聚合消息把结果总结给用户
    messages = await messages_of(conv["id"], role="agent")
    _agent_text(messages, ORCHESTRATOR_ID, "最终总结")


async def test_revise_feedback_reflected_in_next_plan(client: AsyncClient):
    await _ensure_agents()
    conv = await _create_group(client)
    await _drain_pending(conv["id"])

    feedback = "第一步先给目录结构留档"
    async with EventCollector() as collector:
        watcher = asyncio.create_task(
            _watch_reviews(conv["id"], [("revise", feedback), ("approve", None)])
        )
        res = await client.post(
            f"/api/conversations/{conv['id']}/messages",
            json={"content": "整理一份要点文档"},
        )
        assert res.status_code == 202, res.text
        await wait_for_conversation_runs(conv["id"], min_runs=4)
        await watcher

    pendings = collector.of_type("dispatch.plan.pending")
    assert len(pendings) == 2
    # 修改意见回灌后，新计划的任务文本带着反馈
    assert feedback in pendings[1].pendingPlan.plan[0].task
    assert feedback not in pendings[0].pendingPlan.plan[0].task

    # 重排后的计划只审批通过一次（revise 的那份不进执行）
    plans = collector.of_type("dispatch.plan")
    assert len(plans) == 1
    assert feedback in plans[0].plan[0].task

    ends = collector.of_type("dispatch.end")
    assert len(ends) == 3 and all(e.status == "complete" for e in ends)

    resolved = collector.of_type("dispatch.plan.resolved")
    assert [(e.approved, e.revising) for e in resolved] == [(False, True), (True, None)]
