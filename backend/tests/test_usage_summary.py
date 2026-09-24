"""Token 用量聚合（build_usage_summary + GET /api/usage/summary）。

口径：today/week 是滚动窗口（now-24h / now-7d）；totalTokens 含 cache 读写；
byModel 按 usage.model 分组（没有 model 的 run 不进）；top 会话按 totalTokens 取前 10。
"""

from __future__ import annotations

from typing import Any

import pytest_asyncio
from sqlalchemy import delete

from app.db.bootstrap import bootstrap_database
from app.db.models import Agent, AgentRun, Conversation
from app.db.session import SessionLocal
from app.services.usage_service import build_usage_summary
from app.utils.time import now_ms

AG_A = "ag_use_a"
AG_B = "ag_use_b"

_DAY_MS = 24 * 60 * 60 * 1000


@pytest_asyncio.fixture(autouse=True)
async def _fresh_db():
    """建表 + 清掉会影响全局聚合的旧数据（tempdir 是整个 session 共享的）。

    用量聚合是全局查询，别的测试模块留下的带 usage 的 run 会算进 bucket，
    所以这里连「不归本模块」的 usage run 一起清。
    """
    await bootstrap_database()
    async with SessionLocal() as session:
        await session.execute(delete(AgentRun).where(AgentRun.usage.is_not(None)))
        await session.execute(delete(AgentRun).where(AgentRun.id.like("run_use_%")))
        await session.execute(delete(Conversation).where(Conversation.id.like("conv_use_%")))
        await session.execute(delete(Agent).where(Agent.id.in_([AG_A, AG_B])))
        await session.commit()
    yield


async def _add_agent(agent_id: str, name: str) -> None:
    async with SessionLocal() as session:
        session.add(
            Agent(
                id=agent_id,
                name=name,
                avatar="🤖",
                description="",
                capabilities=[],
                system_prompt="",
                adapter_name="mock",
                model_provider=None,
                model_id=None,
                api_key=None,
                api_base_url=None,
                tool_names=[],
                is_builtin=False,
                is_orchestrator=False,
                supports_vision=False,
                created_at=now_ms(),
            )
        )
        await session.commit()


async def _add_conversation(conv_id: str, title: str) -> None:
    async with SessionLocal() as session:
        session.add(
            Conversation(
                id=conv_id,
                title=title,
                mode="single",
                agent_ids=[AG_A],
                pinned_message_ids=[],
                bookmarked_message_ids=[],
                archived=False,
                pinned_at=None,
                fs_write_approval_mode="review",
                created_at=now_ms(),
                updated_at=now_ms(),
            )
        )
        await session.commit()


async def _add_run(
    run_id: str,
    conv_id: str,
    agent_id: str,
    started_at: int,
    usage: dict[str, Any] | None,
) -> None:
    async with SessionLocal() as session:
        session.add(
            AgentRun(
                id=run_id,
                conversation_id=conv_id,
                agent_id=agent_id,
                trigger_message_id=None,
                status="complete",
                error=None,
                parent_run_id=None,
                usage=usage,
                started_at=started_at,
                finished_at=started_at,
            )
        )
        await session.commit()


def _usage(
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_creation: int,
    model: str | None,
) -> dict[str, Any]:
    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "cacheReadTokens": cache_read,
        "cacheCreationTokens": cache_creation,
        "model": model,
    }


# ─── 聚合口径 ─────────────────────────────────────────────


async def test_buckets_use_rolling_windows_and_include_cache_tokens():
    now = now_ms()
    await _add_agent(AG_A, "用量 A")
    await _add_conversation("conv_use_1", "用量会话一")

    # 20 = 10 + 5 + 3 + 2（totalTokens 含 cache 读写）
    await _add_run(
        "run_use_today", "conv_use_1", AG_A, now - _DAY_MS // 2, _usage(10, 5, 3, 2, "m-today")
    )
    await _add_run(
        "run_use_week", "conv_use_1", AG_A, now - 2 * _DAY_MS, _usage(100, 0, 0, 0, "m-week")
    )
    await _add_run(
        "run_use_old", "conv_use_1", AG_A, now - 10 * _DAY_MS, _usage(1, 1, 1, 1, "m-old")
    )
    # usage 为空的 run 不计入任何 bucket
    await _add_run("run_use_none", "conv_use_1", AG_A, now - _DAY_MS // 2, None)

    async with SessionLocal() as session:
        summary = await build_usage_summary(session)

    assert summary["today"] == {
        "inputTokens": 10,
        "outputTokens": 5,
        "cacheReadTokens": 3,
        "cacheCreationTokens": 2,
        "totalTokens": 20,
        "runs": 1,
    }
    assert summary["week"]["totalTokens"] == 120
    assert summary["week"]["runs"] == 2
    assert summary["allTime"]["totalTokens"] == 124
    assert summary["allTime"]["runs"] == 3


async def test_groups_by_model_and_agent_sorted_by_total_tokens():
    now = now_ms()
    await _add_agent(AG_A, "用量 A")
    await _add_agent(AG_B, "用量 B")
    await _add_conversation("conv_use_1", "会话一")
    await _add_conversation("conv_use_2", "会话二")

    await _add_run("run_use_a1", "conv_use_1", AG_A, now, _usage(10, 0, 0, 0, "deepseek-v4-flash"))
    await _add_run("run_use_a2", "conv_use_1", AG_A, now, _usage(1, 1, 1, 1, "deepseek-v4-flash"))
    await _add_run("run_use_b1", "conv_use_2", AG_B, now, _usage(100, 0, 0, 0, "gpt-4o"))
    # 没有 model 的 run 不进 byModel
    await _add_run("run_use_b2", "conv_use_2", AG_B, now, _usage(5, 0, 0, 0, None))

    async with SessionLocal() as session:
        summary = await build_usage_summary(session)

    assert summary["byModel"] == [
        {"model": "gpt-4o", "totalTokens": 100, "runs": 1},
        {"model": "deepseek-v4-flash", "totalTokens": 14, "runs": 2},
    ]
    assert summary["byAgent"] == [
        {"agentId": AG_B, "name": "用量 B", "totalTokens": 105, "runs": 2},
        {"agentId": AG_A, "name": "用量 A", "totalTokens": 14, "runs": 2},
    ]
    assert summary["topConversations"] == [
        {
            "id": "conv_use_2",
            "title": "会话二",
            "totalTokens": 105,
            "runs": 2,
            "updatedAt": summary["topConversations"][0]["updatedAt"],
        },
        {
            "id": "conv_use_1",
            "title": "会话一",
            "totalTokens": 14,
            "runs": 2,
            "updatedAt": summary["topConversations"][1]["updatedAt"],
        },
    ]


async def test_top_conversations_capped_at_ten_sorted_by_tokens():
    now = now_ms()
    await _add_agent(AG_A, "用量 A")
    for i in range(12):
        conv_id = f"conv_use_{i:02d}"
        await _add_conversation(conv_id, f"会话 {i:02d}")
        await _add_run(
            f"run_use_top_{i:02d}", conv_id, AG_A, now, _usage(i + 1, 0, 0, 0, "m")
        )

    async with SessionLocal() as session:
        summary = await build_usage_summary(session)

    tops = summary["topConversations"]
    assert len(tops) == 10
    # totalTokens 递减排序，且最多 10 条（最小的两个被截掉）
    assert [t["totalTokens"] for t in tops] == [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    assert tops[0]["id"] == "conv_use_11"


# ─── GET /api/usage/summary 契约 ───────────────────────────


async def test_usage_summary_route_returns_bare_summary(client):
    now = now_ms()
    async with SessionLocal() as session:
        session.add(
            Agent(
                id=AG_A,
                name="用量 A",
                avatar="🤖",
                description="",
                capabilities=[],
                system_prompt="",
                adapter_name="mock",
                model_provider=None,
                model_id=None,
                api_key=None,
                api_base_url=None,
                tool_names=[],
                is_builtin=False,
                is_orchestrator=False,
                supports_vision=False,
                created_at=now_ms(),
            )
        )
        session.add(
            Conversation(
                id="conv_use_1",
                title="会话一",
                mode="single",
                agent_ids=[AG_A],
                pinned_message_ids=[],
                bookmarked_message_ids=[],
                archived=False,
                pinned_at=None,
                fs_write_approval_mode="review",
                created_at=now_ms(),
                updated_at=now_ms(),
            )
        )
        await session.commit()
    await _add_run("run_use_route", "conv_use_1", AG_A, now, _usage(7, 3, 0, 0, "m-route"))

    res = await client.get("/api/usage/summary")
    assert res.status_code == 200, res.text
    body = res.json()
    # 没有 {ok}/{usage} 外层信封，直接是聚合对象
    assert set(body) == {
        "today",
        "week",
        "allTime",
        "topConversations",
        "byAgent",
        "byModel",
    }
    assert body["today"]["totalTokens"] == 10
    assert body["today"]["cacheReadTokens"] == 0
    assert body["topConversations"][0]["title"] == "会话一"
    assert body["byAgent"][0]["agentId"] == AG_A
    assert body["byModel"][0]["model"] == "m-route"
