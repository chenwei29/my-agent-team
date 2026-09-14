"""`/api/conversations/{id}/messages` 契约测试。"""

from __future__ import annotations

import asyncio

from httpx import AsyncClient
from sqlalchemy import func, select

from app.db.models import Message, Workspace
from app.db.session import SessionLocal
from tests.helpers import wait_for_runs, wait_for_conversation_runs

MESSAGE_CAMEL_KEYS = {
    "conversationId",
    "agentId",
    "parentMessageId",
    "mentionedAgentIds",
    "runId",
    "createdAt",
}


async def _wait_until_running(run_id: str, timeout: float = 5.0) -> None:
    import time

    from app.db.models import AgentRun

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with SessionLocal() as session:
            status = await session.scalar(select(AgentRun.status).where(AgentRun.id == run_id))
        if status == "running":
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} never reached running")


async def _count(model, **filters) -> int:
    """绕过 API 直接查库 —— 用来验证级联删除真的落到了表上。"""
    async with SessionLocal() as session:
        stmt = select(func.count()).select_from(model)
        for column, value in filters.items():
            stmt = stmt.where(getattr(model, column) == value)
        return await session.scalar(stmt) or 0


async def test_send_message_returns_202_with_run_ids(client: AsyncClient, conversation: dict):
    res = await client.post(
        f"/api/conversations/{conversation['id']}/messages", json={"content": "你好"}
    )
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["messageId"].startswith("msg_")
    # P2 起：会话里的 mock agent 会被触发，立刻拿到 runId（不等待 run 结束）
    assert len(body["runIds"]) == 1 and body["runIds"][0].startswith("run_")
    # 前端类型是 messages?: MessageRow[]，不能多一个显式 null
    assert "messages" not in body
    await wait_for_runs(body["runIds"])


async def test_sent_message_is_persisted_and_camel_case(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    await client.post(f"/api/conversations/{cid}/messages", json={"content": "一条消息"})

    res = await client.get(f"/api/conversations/{cid}/messages")
    assert res.status_code == 200
    messages = res.json()["messages"]
    assert len(messages) == 1

    msg = messages[0]
    assert msg["id"].startswith("msg_")
    assert msg["conversationId"] == cid
    assert msg["role"] == "user"
    assert msg["agentId"] is None
    assert msg["status"] == "complete"
    assert msg["parts"] == [{"type": "text", "content": "一条消息"}]
    assert MESSAGE_CAMEL_KEYS <= set(msg)
    assert isinstance(msg["createdAt"], int) and msg["createdAt"] > 10**12


async def test_blank_content_creates_message_with_no_parts(client: AsyncClient, conversation: dict):
    """只有 content 去空白后非空才写进 parts；空白 body 仍算合法（有 attachmentIds 时）。"""
    cid = conversation["id"]
    res = await client.post(
        f"/api/conversations/{cid}/messages", json={"content": "   ", "attachmentIds": ["att_x"]}
    )
    assert res.status_code == 202

    messages = (await client.get(f"/api/conversations/{cid}/messages")).json()["messages"]
    assert messages[0]["parts"] == []


async def test_rejects_message_without_content_or_attachments(
    client: AsyncClient, conversation: dict
):
    res = await client.post(f"/api/conversations/{conversation['id']}/messages", json={})
    assert res.status_code == 400


async def test_send_message_to_unknown_conversation_is_400(client: AsyncClient):
    res = await client.post("/api/conversations/conv_nope/messages", json={"content": "x"})
    assert res.status_code == 400


async def test_list_messages_of_unknown_conversation_is_empty_200(client: AsyncClient):
    """这个端点不校验会话存在性 —— 会话不存在也 200 + 空数组，不是 404。"""
    res = await client.get("/api/conversations/conv_nope/messages")
    assert res.status_code == 200
    assert res.json()["messages"] == []


async def test_messages_are_ordered_by_created_at(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    run_ids = []
    for text in ("第一条", "第二条", "第三条"):
        res = await client.post(f"/api/conversations/{cid}/messages", json={"content": text})
        run_ids += res.json()["runIds"]
    await wait_for_runs(run_ids)

    messages = (await client.get(f"/api/conversations/{cid}/messages")).json()["messages"]
    user_messages = [m for m in messages if m["role"] == "user"]
    assert [m["parts"][0]["content"] for m in user_messages] == ["第一条", "第二条", "第三条"]
    assert [m["createdAt"] for m in messages] == sorted(m["createdAt"] for m in messages)


async def test_send_message_bumps_conversation_updated_at(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    before = conversation["updatedAt"]

    await asyncio.sleep(0.01)
    res = await client.post(f"/api/conversations/{cid}/messages", json={"content": "顶上去"})
    await wait_for_runs(res.json()["runIds"])

    listed = (await client.get("/api/conversations")).json()["conversations"]
    after = next(c for c in listed if c["id"] == cid)["updatedAt"]
    assert after > before


async def test_mentioned_agent_ids_round_trip(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    res = await client.post(
        f"/api/conversations/{cid}/messages",
        json={"content": "点名", "mentionedAgentIds": ["ag_pm"]},
    )
    await wait_for_runs(res.json()["runIds"])
    msg = (await client.get(f"/api/conversations/{cid}/messages")).json()["messages"][0]
    assert msg["mentionedAgentIds"] == ["ag_pm"]


async def test_clear_history_keeps_conversation(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    run_ids = []
    for text in ("a", "b"):
        res = await client.post(f"/api/conversations/{cid}/messages", json={"content": text})
        run_ids += res.json()["runIds"]
    await wait_for_runs(run_ids)

    res = await client.delete(f"/api/conversations/{cid}/messages")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["deletedMessageCount"] == 4  # 2 条用户消息 + 2 条 mock 回复
    assert body["deletedRunCount"] == 2
    assert body["deletedSummaryCount"] == 0
    # 会话本身还在，且消息已空
    assert body["conversation"]["id"] == cid
    assert (await client.get(f"/api/conversations/{cid}/messages")).json()["messages"] == []


async def test_clear_history_refuses_while_a_run_is_active(client: AsyncClient, conversation: dict):
    """run 还在跑时清空历史 → 409（错误文案是 Cannot clear conversation history while...）。"""
    cid = conversation["id"]
    res = await client.post(f"/api/conversations/{cid}/messages", json={"content": "你好"})
    run_id = res.json()["runIds"][0]

    # 等 run 真的起来（POST 返回时后台任务可能还没落 run 行，直接删会误判成"没有活跃 run"）
    await _wait_until_running(run_id)

    res = await client.delete(f"/api/conversations/{cid}/messages")
    assert res.status_code == 409, res.text
    assert "agent runs are active" in res.json()["error"]

    await wait_for_conversation_runs(cid)
    assert (await client.delete(f"/api/conversations/{cid}/messages")).status_code == 200


async def test_clear_history_on_unknown_conversation_is_404(client: AsyncClient):
    res = await client.delete("/api/conversations/conv_nope/messages")
    assert res.status_code == 404


async def test_delete_conversation_cascades_messages_and_workspace(
    client: AsyncClient, conversation: dict
):
    """必须开 PRAGMA foreign_keys=ON 这条才过：否则 messages / workspaces 会留孤儿行。"""
    cid = conversation["id"]
    res = await client.post(f"/api/conversations/{cid}/messages", json={"content": "级联"})
    await wait_for_runs(res.json()["runIds"])
    assert await _count(Message, conversation_id=cid) == 2  # 用户消息 + mock 回复
    assert await _count(Workspace, conversation_id=cid) == 1

    assert (await client.delete(f"/api/conversations/{cid}")).status_code == 200

    assert await _count(Message, conversation_id=cid) == 0
    assert await _count(Workspace, conversation_id=cid) == 0
