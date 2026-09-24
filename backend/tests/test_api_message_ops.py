"""消息高级操作：withdraw / edit / pin / bookmark / regenerate。"""

from __future__ import annotations

import asyncio

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select

from app.db.models import AgentRun, Artifact, Conversation, Message
from app.db.session import SessionLocal
from app.tools.artifacts import WRITE_ARTIFACT_TOOL
from app.tools.types import ToolContext
from tests.helpers import EventCollector, wait_for_conversation_runs, wait_for_runs

MOCK_ID = "ag_e2e_mock"


@pytest.fixture
async def conversation(client: AsyncClient) -> dict:
    res = await client.post("/api/conversations", json={"mode": "single", "agentIds": [MOCK_ID]})
    assert res.status_code == 201, res.text
    return res.json()["conversation"]


async def _send(client: AsyncClient, conversation_id: str, content: str) -> str:
    res = await client.post(
        f"/api/conversations/{conversation_id}/messages", json={"content": content}
    )
    assert res.status_code == 202, res.text
    await wait_for_runs(res.json()["runIds"])
    return res.json()["messageId"]


async def _rows(conversation_id: str) -> list[Message]:
    async with SessionLocal() as session:
        return list(
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
            )
        )


async def _load_conversation(conversation_id: str) -> Conversation:
    async with SessionLocal() as session:
        return await session.get(Conversation, conversation_id)


async def _count(model, **filters) -> int:
    async with SessionLocal() as session:
        stmt = select(func.count()).select_from(model)
        for column, value in filters.items():
            stmt = stmt.where(getattr(model, column) == value)
        return await session.scalar(stmt) or 0


# ─── pin / bookmark ────────────────────────────────────────


async def test_bookmark_toggle(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    message_id = await _send(client, cid, "书签我")
    before = await _load_conversation(cid)

    res = await client.post(f"/api/messages/{message_id}/bookmark", json={"conversationId": cid})
    assert res.status_code == 200
    assert res.json() == {"bookmarkedMessageIds": [message_id], "bookmarked": True}

    after = await _load_conversation(cid)
    assert after.bookmarked_message_ids == [message_id]
    assert before.bookmarked_message_ids == []
    # 书签算「会话活跃」，会更新 updatedAt
    assert after.updated_at >= before.updated_at

    res = await client.post(f"/api/messages/{message_id}/bookmark", json={"conversationId": cid})
    assert res.json() == {"bookmarkedMessageIds": [], "bookmarked": False}


async def test_pin_toggle_and_limit(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    message_ids = [await _send(client, cid, f"第 {i} 条") for i in range(6)]

    before = await _load_conversation(cid)
    for message_id in message_ids[:5]:
        res = await client.post(f"/api/messages/{message_id}/pin", json={"conversationId": cid})
        assert res.status_code == 200, res.text
    assert res.json() == {"pinnedMessageIds": message_ids[:5], "pinned": True}

    # 第 6 条超上限 → 400 PIN_LIMIT_EXCEEDED
    res = await client.post(f"/api/messages/{message_ids[5]}/pin", json={"conversationId": cid})
    assert res.status_code == 400
    assert res.json()["error"] == "PIN_LIMIT_EXCEEDED"

    # 取消一条后又能再 pin
    res = await client.post(f"/api/messages/{message_ids[0]}/pin", json={"conversationId": cid})
    assert res.json() == {"pinnedMessageIds": message_ids[1:5], "pinned": False}
    res = await client.post(f"/api/messages/{message_ids[5]}/pin", json={"conversationId": cid})
    assert res.status_code == 200

    after = await _load_conversation(cid)
    # pin 不算会话活跃，不动 updatedAt
    assert after.updated_at == before.updated_at


async def test_pin_and_bookmark_errors_are_400(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    res = await client.post("/api/messages/msg_nope/pin", json={"conversationId": cid})
    assert res.status_code == 400
    assert res.json()["error"] == "Message not found in conversation: msg_nope"

    res = await client.post(
        "/api/messages/msg_nope/bookmark", json={"conversationId": "conv_nope"}
    )
    assert res.status_code == 400
    assert "Conversation not found" in res.json()["error"]

    # 别的会话的消息不能被本会话 pin
    other = await client.post(
        "/api/conversations", json={"mode": "single", "agentIds": [MOCK_ID]}
    )
    other_cid = other.json()["conversation"]["id"]
    message_id = await _send(client, other_cid, "别处的消息")
    res = await client.post(f"/api/messages/{message_id}/pin", json={"conversationId": cid})
    assert res.status_code == 400
    assert "not found" in res.json()["error"]


# ─── withdraw ──────────────────────────────────────────────


async def test_withdraw_deletes_window_and_artifacts(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    message_id = await _send(client, cid, "产出一个文档")

    result = await WRITE_ARTIFACT_TOOL.handler(
        {"type": "document", "title": "件", "content": {"format": "markdown", "content": "x"}},
        ToolContext(
            conversation_id=cid,
            agent_id=MOCK_ID,
            run_id="run_test",
            workspace_path=".",
            abort_signal=None,
        ),
    )
    assert result.ok
    artifact_id = result.value["artifactId"]

    async with EventCollector() as collector:
        res = await client.post(f"/api/messages/{message_id}/withdraw", json={"conversationId": cid})
        await asyncio.sleep(0.05)  # 等事件被消费

    assert res.status_code == 200, res.text
    body = res.json()
    assert message_id in body["deletedMessageIds"]
    assert body["deletedArtifactIds"] == []

    # 时间窗 >= 触发消息：agent 的回复也一起走
    assert await _count(Message, conversation_id=cid) == 0
    assert await _count(AgentRun, conversation_id=cid) == 0

    removed = collector.of_type("message.removed")
    assert len(removed) == 1
    assert message_id in removed[0].messageIds

    # 产物只在被 artifact_ref 引用时才级联删；这里没引用，仍在库里
    assert await _count(Artifact, id=artifact_id) == 1


async def test_withdraw_cascades_referenced_artifacts(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    message_id = await _send(client, cid, "带产物的消息")

    # 手工把 artifact_ref part 挂到这条 user 消息上（正常由 adapter/runner 注入）
    async with SessionLocal() as session:
        row = await session.get(Message, message_id)
        row.parts = [
            *row.parts,
            {"type": "artifact_ref", "artifactId": "art_ref_1", "title": "件", "artifactType": "document"},
        ]
        session.add(
            Artifact(
                id="art_ref_1",
                conversation_id=cid,
                type="document",
                title="件",
                content={"type": "document", "format": "markdown", "content": "x"},
                version=1,
                parent_artifact_id=None,
                created_by_agent_id=MOCK_ID,
                created_at=row.created_at,
            )
        )
        await session.commit()

    res = await client.post(f"/api/messages/{message_id}/withdraw", json={"conversationId": cid})
    assert res.status_code == 200
    assert res.json()["deletedArtifactIds"] == ["art_ref_1"]
    assert await _count(Artifact, id="art_ref_1") == 0


async def test_withdraw_only_latest_user_message(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    first = await _send(client, cid, "第一条")
    await _send(client, cid, "第二条")

    res = await client.post(f"/api/messages/{first}/withdraw", json={"conversationId": cid})
    assert res.status_code == 400
    assert res.json()["error"] == "Only the latest user message can be withdrawn"

    # agent 消息不能撤回
    rows = await _rows(cid)
    agent_message = next(r for r in rows if r.role == "agent")
    res = await client.post(f"/api/messages/{agent_message.id}/withdraw", json={"conversationId": cid})
    assert res.status_code == 400
    assert res.json()["error"] == "Only user messages can be withdrawn"

    res = await client.post("/api/messages/msg_nope/withdraw", json={"conversationId": cid})
    assert res.status_code == 404
    assert res.json()["error"] == "Message not found: msg_nope"


# ─── regenerate / edit ─────────────────────────────────────


async def test_regenerate_keeps_user_message(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    message_id = await _send(client, cid, "再答一次")

    res = await client.post(f"/api/conversations/{cid}/regenerate")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["triggerMessageId"] == message_id
    assert len(body["runIds"]) == 1
    # 只删触发消息之后的回复，user 消息本身保留
    assert message_id not in body["deletedMessageIds"]
    assert len(body["deletedMessageIds"]) >= 1

    rows = await _rows(cid)
    assert rows[0].id == message_id

    await wait_for_runs(body["runIds"])

    empty = await client.post(
        "/api/conversations/conv_nope/regenerate"
    )
    assert empty.status_code == 400


async def test_regenerate_without_user_message(client: AsyncClient, conversation: dict):
    res = await client.post(f"/api/conversations/{conversation['id']}/regenerate")
    assert res.status_code == 400
    assert res.json()["error"] == "No user message to regenerate from"


async def test_edit_resends_with_new_content(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    message_id = await _send(client, cid, "原始内容")

    res = await client.post(
        f"/api/messages/{message_id}/edit",
        json={"conversationId": cid, "content": "  改过的内容  "},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert message_id in body["deletedMessageIds"]
    assert len(body["runIds"]) == 1

    new_message = body["newMessage"]
    assert new_message["id"] != message_id
    assert new_message["conversationId"] == cid
    assert new_message["parts"] == [{"type": "text", "content": "改过的内容"}]
    assert new_message["mentionedAgentIds"] == []

    rows = await _rows(cid)
    assert rows[0].id == new_message["id"]
    await wait_for_conversation_runs(cid)

    # agent 消息不能编辑
    agent_message = next(r for r in await _rows(cid) if r.role == "agent")
    res = await client.post(
        f"/api/messages/{agent_message.id}/edit",
        json={"conversationId": cid, "content": "x"},
    )
    assert res.status_code == 400
    assert res.json()["error"] == "Only user messages can be edited"

    res = await client.post(
        "/api/messages/msg_nope/edit", json={"conversationId": cid, "content": "x"}
    )
    assert res.status_code == 404
