"""AgentRunner / SSE 流水线测试（P2 的核心）。

覆盖：起 run、边消费边落库、事件顺序、先落库再推流、中止、群聊多 run 并发、
没有 adapter 的 agent 明确失败（不假装成功）。
"""

from __future__ import annotations

import asyncio
import json

from httpx import AsyncClient
from sqlalchemy import select

from app.api import stream as stream_module
from app.api.stream import _event_stream
from app.db.bootstrap_cli import E2E_MOCK_AGENT_ID
from app.db.models import Agent, Message
from app.db.session import SessionLocal
from app.services.conversation_service import decide_responders
from app.services.event_bus import event_bus
from app.utils.time import now_ms
from tests.helpers import (
    EventCollector,
    messages_of,
    run_row,
    wait_for_conversation_runs,
    wait_for_runs,
)

MOCK_AGENT_ID = E2E_MOCK_AGENT_ID
SECOND_MOCK_AGENT_ID = "ag_e2e_mock_2"


class _FakeRequest:
    """只实现 SSE 生成器用到的那一个方法。"""

    async def is_disconnected(self) -> bool:
        return False


async def _ensure_second_mock_agent() -> str:
    async with SessionLocal() as session:
        existing = await session.scalar(select(Agent).where(Agent.id == SECOND_MOCK_AGENT_ID))
        if existing is None:
            session.add(
                Agent(
                    id=SECOND_MOCK_AGENT_ID,
                    name="E2E Mock 2",
                    avatar="🤖",
                    description="第二个 mock agent（群聊并发测试用）",
                    capabilities=["test"],
                    system_prompt="mock",
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
    return SECOND_MOCK_AGENT_ID


async def _create_conversation(client: AsyncClient, agent_ids: list[str], mode: str) -> dict:
    res = await client.post(
        "/api/conversations", json={"mode": mode, "agentIds": agent_ids}
    )
    assert res.status_code == 201, res.text
    return res.json()["conversation"]


# ─── 起 run + 落库 ──────────────────────────────────────────


async def test_send_message_starts_a_mock_run(client: AsyncClient, conversation: dict):
    res = await client.post(
        f"/api/conversations/{conversation['id']}/messages", json={"content": "你好"}
    )
    assert res.status_code == 202, res.text
    run_ids = res.json()["runIds"]
    assert len(run_ids) == 1 and run_ids[0].startswith("run_")

    await wait_for_runs(run_ids)
    row = await run_row(run_ids[0])
    assert row is not None
    assert row.status == "complete"
    assert row.error is None
    assert isinstance(row.started_at, int) and row.started_at > 10**12
    assert isinstance(row.finished_at, int) and row.finished_at >= row.started_at
    assert row.trigger_message_id == res.json()["messageId"]


async def test_streamed_reply_is_persisted_with_parts(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    res = await client.post(f"/api/conversations/{cid}/messages", json={"content": "你好"})
    run_id = res.json()["runIds"][0]
    await wait_for_runs([run_id])

    messages = (await client.get(f"/api/conversations/{cid}/messages")).json()["messages"]
    assert [m["role"] for m in messages] == ["user", "agent"]

    agent_message = messages[1]
    assert agent_message["runId"] == run_id
    assert agent_message["status"] == "complete"
    assert agent_message["agentId"] == MOCK_AGENT_ID

    types = [p["type"] for p in agent_message["parts"]]
    assert types == ["thinking", "text"]
    text = agent_message["parts"][1]["content"]
    assert "我是 Mock Agent" in text  # 刷新后内容仍在 = 真的落库了，不是内存假象
    assert "\n\n" in text


async def test_code_reply_persists_code_part_with_language(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    res = await client.post(f"/api/conversations/{cid}/messages", json={"content": "写代码"})
    await wait_for_runs(res.json()["runIds"])

    agent_message = (await messages_of(cid, role="agent"))[0]
    code_parts = [p for p in agent_message.parts if p["type"] == "code"]
    assert len(code_parts) == 1
    assert code_parts[0]["language"] == "tsx"
    assert "export function Counter()" in code_parts[0]["content"]


# ─── 事件顺序 / 先落库再推流 ────────────────────────────────


async def test_event_order_message_added_then_run_start_then_stream_then_run_end(
    client: AsyncClient, conversation: dict
):
    cid = conversation["id"]
    async with EventCollector() as collector:
        res = await client.post(f"/api/conversations/{cid}/messages", json={"content": "你好"})
        await wait_for_runs(res.json()["runIds"])
        await asyncio.sleep(0.05)

    types = collector.types()
    assert types[0] == "message.added"
    assert types[1] == "run.start"
    assert types[2] == "message.start"
    assert types[-1] == "run.end"
    assert types.index("message.end") < types.index("run.end")

    deltas = [e for e in collector.events if e.type == "part.delta"]
    assert len(deltas) >= 20  # 打字机：每条 delta 只有 4/8 个字
    assert all(e.delta.text for e in deltas)

    run_end = collector.of_type("run.end")[0]
    assert run_end.status == "complete"
    assert run_end.error is None


async def test_db_write_happens_before_event_publish(client: AsyncClient, conversation: dict):
    """收到 message.end 的那一刻，库里该消息必须已经是 complete（客户端可能立刻重新拉取）。"""
    observed: list[str] = []
    done = asyncio.Event()

    async def consume() -> None:
        async for event in event_bus.subscribe():
            if event.type == "message.end":
                async with SessionLocal() as session:
                    row = await session.scalar(select(Message).where(Message.id == event.messageId))
                observed.append(row.status if row else "missing")
                done.set()

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    try:
        await client.post(f"/api/conversations/{conversation['id']}/messages", json={"content": "你好"})
        await asyncio.wait_for(done.wait(), timeout=10)
    finally:
        task.cancel()

    assert observed == ["complete"]


# ─── 中止 ───────────────────────────────────────────────────


async def test_abort_marks_run_aborted_and_appends_marker(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    async with EventCollector() as collector:
        res = await client.post(f"/api/conversations/{cid}/messages", json={"content": "你好"})
        run_id = res.json()["runIds"][0]

        abort = await client.post(f"/api/runs/{run_id}/abort")
        assert abort.status_code == 200 and abort.json() == {"ok": True}

        await wait_for_runs([run_id])
        await asyncio.sleep(0.05)

    row = await run_row(run_id)
    assert row.status == "aborted"
    assert row.error is None  # 中止不带 error（run.end.error 同样为空）

    run_end = collector.of_type("run.end")[0]
    assert run_end.status == "aborted"
    assert run_end.error is None

    agent_message = (await messages_of(cid, role="agent"))[0]
    assert agent_message.parts[-1] == {"type": "text", "content": "[已中止]"}
    # 被中止 → 正文没吐完
    text_parts = [p for p in agent_message.parts if p["type"] == "text"]
    assert all("我是 Mock Agent" not in p["content"] for p in text_parts)


async def test_abort_finished_run_is_404(client: AsyncClient, conversation: dict):
    res = await client.post(
        f"/api/conversations/{conversation['id']}/messages", json={"content": "你好"}
    )
    run_id = res.json()["runIds"][0]
    await wait_for_runs([run_id])

    abort = await client.post(f"/api/runs/{run_id}/abort")
    assert abort.status_code == 404
    assert abort.json()["error"] == "Run not found or already finished"


async def test_abort_unknown_run_is_404(client: AsyncClient):
    res = await client.post("/api/runs/run_nope/abort")
    assert res.status_code == 404


# ─── 群聊并发 ───────────────────────────────────────────────


async def test_group_mention_starts_two_concurrent_runs(client: AsyncClient):
    second = await _ensure_second_mock_agent()
    conversation = await _create_conversation(
        client, [MOCK_AGENT_ID, second], mode="group"
    )
    cid = conversation["id"]

    res = await client.post(
        f"/api/conversations/{cid}/messages",
        json={"content": "你好", "mentionedAgentIds": [MOCK_AGENT_ID, second]},
    )
    run_ids = res.json()["runIds"]
    assert len(run_ids) == 2

    # 两个 run 同时还在跑 → 证明并发而不是串行
    await asyncio.sleep(0.1)
    running = [r for r in await asyncio.gather(*(run_row(r) for r in run_ids)) if r.status == "running"]
    assert len(running) == 2

    await wait_for_runs(run_ids)
    rows = await asyncio.gather(*(run_row(r) for r in run_ids))
    assert [r.status for r in rows] == ["complete", "complete"]

    agent_messages = await messages_of(cid, role="agent")
    assert len(agent_messages) == 2
    assert {m.agent_id for m in agent_messages} == {MOCK_AGENT_ID, second}


async def test_group_without_mention_only_triggers_orchestrator(client: AsyncClient):
    conversation = await _create_conversation(
        client, ["ag_orchestrator", "ag_pm"], mode="group"
    )
    res = await client.post(
        f"/api/conversations/{conversation['id']}/messages", json={"content": "帮我做个东西"}
    )
    run_ids = res.json()["runIds"]
    assert len(run_ids) == 1
    await wait_for_runs(run_ids)
    row = await run_row(run_ids[0])
    assert row.agent_id == "ag_orchestrator"


async def test_group_mention_outside_conversation_starts_nothing(client: AsyncClient):
    conversation = await _create_conversation(client, [MOCK_AGENT_ID], mode="single")
    res = await client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={"content": "你好", "mentionedAgentIds": ["ag_not_in_conv"]},
    )
    # 单聊忽略 @：仍然起 mock agent 的 run
    assert len(res.json()["runIds"]) == 1
    await wait_for_runs(res.json()["runIds"])


# ─── 配置缺失的 agent 要明确失败 ────────────────────────


async def test_custom_agent_without_key_fails_loudly(client: AsyncClient):
    """custom agent（deepseek）没有任何可用 key：run 必须明确失败，不能假装成功。"""
    conversation = await _create_conversation(client, ["ag_pm"], mode="single")
    cid = conversation["id"]

    res = await client.post(f"/api/conversations/{cid}/messages", json={"content": "写个 PRD"})
    run_id = res.json()["runIds"][0]
    await wait_for_runs([run_id])

    row = await run_row(run_id)
    assert row.status == "failed"
    assert "DEEPSEEK_API_KEY not set" in row.error

    messages = await messages_of(cid)
    error_message = [m for m in messages if m.id.startswith("msg_err_")][0]
    assert error_message.status == "error"
    assert error_message.parts[0]["content"].startswith("[失败]")
    assert "DEEPSEEK_API_KEY not set" in error_message.parts[0]["content"]


# ─── decide_responders 纯函数 ───────────────────────────────


class _Conv:
    def __init__(self, mode: str, agent_ids: list[str]) -> None:
        self.mode = mode
        self.agent_ids = agent_ids


class _Agent:
    def __init__(self, agent_id: str, is_orchestrator: bool = False) -> None:
        self.id = agent_id
        self.is_orchestrator = is_orchestrator


def test_decide_responders_single_returns_all_members():
    conv = _Conv("single", ["ag_a", "ag_b"])
    assert decide_responders(conv, [], []) == ["ag_a", "ag_b"]  # type: ignore[arg-type]


def test_decide_responders_group_with_mention_filters_and_keeps_order():
    conv = _Conv("group", ["ag_a", "ag_b", "ag_c"])
    assert decide_responders(conv, ["ag_c", "ag_a", "ag_x"], []) == ["ag_c", "ag_a"]  # type: ignore[arg-type]


def test_decide_responders_group_without_mention_picks_the_orchestrator():
    conv = _Conv("group", ["ag_a", "ag_b"])
    agents = [_Agent("ag_a"), _Agent("ag_b", is_orchestrator=True)]
    assert decide_responders(conv, [], agents) == ["ag_b"]  # type: ignore[arg-type]


def test_decide_responders_group_without_orchestrator_starts_nothing():
    conv = _Conv("group", ["ag_a", "ag_b"])
    assert decide_responders(conv, [], [_Agent("ag_a"), _Agent("ag_b")]) == []  # type: ignore[arg-type]


# ─── 端到端：POST → runner → bus → SSE ─────────────────────


async def test_sse_receives_the_whole_run(client: AsyncClient, conversation: dict, monkeypatch):
    """一条真实流水线：HTTP 发消息 → 后台 run → 事件总线 → SSE 帧。"""
    monkeypatch.setattr(stream_module, "HEARTBEAT_SECONDS", 5)
    generator = _event_stream(_FakeRequest())  # type: ignore[arg-type]

    payloads: list[dict] = []

    async def pump() -> None:
        while True:
            try:
                frame = await asyncio.wait_for(anext(generator), timeout=15)
            except (TimeoutError, StopAsyncIteration):
                return
            payload = json.loads(frame[len("data: ") :].strip())
            payloads.append(payload)
            if payload["type"] == "run.end":
                return

    # 先把消费端挂上（生成器先发 connected 再订阅，晚订阅会丢早期事件）
    task = asyncio.create_task(pump())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    try:
        await client.post(
            f"/api/conversations/{conversation['id']}/messages", json={"content": "你好"}
        )
        await asyncio.wait_for(task, timeout=20)
    finally:
        await generator.aclose()

    types = [p["type"] for p in payloads]
    assert types[0] == "connected"
    assert types[1] == "message.added"
    assert types[2] == "run.start"
    assert "part.delta" in types
    assert "message.end" in types
    assert types[-1] == "run.end"
    assert payloads[-1]["status"] == "complete"
