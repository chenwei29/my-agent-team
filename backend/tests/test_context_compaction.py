"""上下文压缩（compact_conversation + 历史注入摘要 + prompt 前缀）。

摘要模型在测试里用假实现替换（不打真实 LLM）；启发式兜底路径单独测一条。
历史注入的「摘要 + 覆盖点之后的消息」语义在 build_history_for 的用例里锁住。
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from app.db.bootstrap import bootstrap_database
from app.db.models import Agent, Artifact, ContextSummary, Conversation, Message
from app.db.session import SessionLocal
from app.errors import ServiceError
from app.utils.ids import new_message_id
from app.services.context_compaction import (
    SummaryModelChoice,
    compact_conversation,
    prefix_prompt_with_context_summary,
)
from app.services.conversation_context import build_history_for
from app.utils.model_registry import estimate_tokens
from app.utils.time import now_ms

AG = "ag_cpx"
CONV = "conv_cpx"

# 消息时间戳逐条 +1，保证排序确定（同一毫秒内插入的多条消息顺序不稳定）
_CLOCK = [now_ms()]


@pytest_asyncio.fixture(autouse=True)
async def _fresh_db():
    """建表 + 清掉本模块固定 id 的旧数据（tempdir 是整个 session 共享的）。"""
    await bootstrap_database()
    async with SessionLocal() as session:
        await session.execute(delete(ContextSummary).where(ContextSummary.conversation_id == CONV))
        await session.execute(delete(Message).where(Message.conversation_id == CONV))
        await session.execute(delete(Artifact).where(Artifact.conversation_id == CONV))
        await session.execute(delete(Conversation).where(Conversation.id == CONV))
        await session.execute(delete(Agent).where(Agent.id == AG))
        await session.commit()
    yield


async def _setup() -> None:
    async with SessionLocal() as session:
        session.add(
            Agent(
                id=AG,
                name="压缩测试 Agent",
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
                id=CONV,
                title="压缩测试会话",
                mode="single",
                agent_ids=[AG],
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


async def _add_message(
    msg_id: str,
    role: str = "user",
    content: str = "",
    *,
    status: str = "complete",
    parts: list[dict[str, Any]] | None = None,
) -> str:
    _CLOCK[0] += 1
    async with SessionLocal() as session:
        session.add(
            Message(
                id=msg_id,
                conversation_id=CONV,
                role=role,
                agent_id=None,
                parts=parts if parts is not None else [{"type": "text", "content": content}],
                status=status,
                parent_message_id=None,
                mentioned_agent_ids=[],
                run_id=None,
                usage=None,
                created_at=_CLOCK[0],
            )
        )
        await session.commit()
    return msg_id


async def _add_n_messages(prefix: str, count: int, *, content_prefix: str = "第") -> list[str]:
    ids = []
    for i in range(1, count + 1):
        msg_id = f"{prefix}{i}"
        await _add_message(msg_id, content=f"{content_prefix}{i} 条历史消息")
        ids.append(msg_id)
    return ids


async def _pin(msg_ids: list[str]) -> None:
    async with SessionLocal() as session:
        conv = await session.scalar(select(Conversation).where(Conversation.id == CONV))
        conv.pinned_message_ids = list(msg_ids)
        await session.commit()


def _fake_choice(
    text: str = "压缩后的摘要内容", captured: list[str] | None = None
) -> SummaryModelChoice:
    async def summarize(prompt: str) -> str:
        if captured is not None:
            captured.append(prompt)
        return text

    return SummaryModelChoice(provider="fake", model_id="fake-model", summarize=summarize)


def _patch_choice(monkeypatch: pytest.MonkeyPatch, choice: SummaryModelChoice) -> None:
    async def _choose(_session, _agent_ids):
        return choice

    monkeypatch.setattr("app.services.context_compaction._choose_summary_model", _choose)


# ─── 压缩流程 ─────────────────────────────────────────────


async def test_compact_without_history_raises(monkeypatch):
    await _setup()
    _patch_choice(monkeypatch, _fake_choice())
    with pytest.raises(ServiceError, match="No compactable history yet"):
        async with SessionLocal() as session:
            await compact_conversation(session, CONV)


async def test_compact_keeps_recent_six_and_summarizes_older(monkeypatch):
    await _setup()
    await _add_n_messages("msg_cpx_", 10)
    _patch_choice(monkeypatch, _fake_choice())

    async with SessionLocal() as session:
        result = await compact_conversation(session, CONV)

    summary = result["summary"]
    message = result["message"]
    assert summary.source_message_count == 4  # 10 条压缩掉 4 条，最近 6 条保留
    assert summary.covered_until_message_id == "msg_cpx_4"
    assert summary.token_estimate == estimate_tokens("压缩后的摘要内容")
    assert summary.model_provider == "fake"
    assert summary.model_id == "fake-model"
    assert summary.id.startswith("ctx_")

    assert message.role == "system"
    assert message.parts == [
        {"type": "text", "content": "已压缩早期上下文，覆盖 4 条消息。"}
    ]
    assert message.status == "complete"


async def test_compact_skips_pinned_and_system_messages(monkeypatch):
    await _setup()
    ids = await _add_n_messages("msg_cpx_", 10)
    await _pin([ids[1]])  # m2 pin 住
    await _add_message("msg_cpx_sys", role="system", content="系统提示消息")
    _patch_choice(monkeypatch, _fake_choice())

    async with SessionLocal() as session:
        result = await compact_conversation(session, CONV)

    # 可压缩池 = 9 条（10 条里去掉 pinned 的 m2；system 不进池）→ 保留最近 6 → 压缩 m1、m3、m4
    summary = result["summary"]
    assert summary.source_message_count == 3
    assert summary.covered_until_message_id == "msg_cpx_4"
    assert summary.covered_until_message_id != "msg_cpx_2"


async def test_compact_bumps_conversation_updated_at(monkeypatch):
    await _setup()
    await _add_n_messages("msg_cpx_", 8)
    _patch_choice(monkeypatch, _fake_choice())

    async with SessionLocal() as session:
        result = await compact_conversation(session, CONV)
    async with SessionLocal() as session:
        conv = await session.scalar(select(Conversation).where(Conversation.id == CONV))

    assert conv.updated_at == result["summary"].created_at


async def test_empty_summary_from_model_raises(monkeypatch):
    await _setup()
    await _add_n_messages("msg_cpx_", 8)
    _patch_choice(monkeypatch, _fake_choice(text="   "))
    with pytest.raises(ServiceError, match="empty summary"):
        async with SessionLocal() as session:
            await compact_conversation(session, CONV)


async def test_heuristic_summary_when_no_key(monkeypatch):
    """没有可用 key 时走本地兜底摘要：provider/modelId 记 null，摘要文本含兜底说明。"""
    await _setup()
    await _add_n_messages("msg_cpx_", 8)

    async def _no_key(_session, _provider):
        return None

    monkeypatch.setattr("app.services.context_compaction.get_effective_api_key", _no_key)

    async with SessionLocal() as session:
        result = await compact_conversation(session, CONV)

    summary = result["summary"]
    assert summary.model_provider is None
    assert summary.model_id is None
    assert "本摘要由本地兜底规则生成" in summary.summary
    # 兜底摘要把渲染后的历史（<messages_to_compact>）也带上了
    assert "<messages_to_compact>" in summary.summary
    assert "第1 条历史消息" in summary.summary


async def test_second_compact_references_previous_summary(monkeypatch):
    await _setup()
    await _add_n_messages("msg_cpx_", 8)
    _patch_choice(monkeypatch, _fake_choice())
    async with SessionLocal() as session:
        first = await compact_conversation(session, CONV)

    await _add_n_messages("msg_cpx_b_", 8, content_prefix="追加")
    captured: list[str] = []
    _patch_choice(monkeypatch, _fake_choice(captured=captured))
    async with SessionLocal() as session:
        second = await compact_conversation(session, CONV)

    assert second["summary"].covered_until_created_at > first["summary"].covered_until_created_at
    assert "<previous_summary>" in captured[0]
    assert "压缩后的摘要内容" in captured[0]
    assert "追加1 条历史消息" in captured[0]
    # 第一次压缩覆盖点之前的消息不再进第二次的压缩输入
    assert "第1 条历史消息" not in captured[0]


# ─── 历史注入：摘要 + 摘要之后的消息 ─────────────────────────


async def test_history_prefers_summary_after_compact(monkeypatch):
    await _setup()
    await _add_n_messages("msg_cpx_", 10)
    _patch_choice(monkeypatch, _fake_choice())
    async with SessionLocal() as session:
        await compact_conversation(session, CONV)

    async with SessionLocal() as session:
        history = await build_history_for(session, AG, CONV)

    contents = [m["content"] for m in history]
    assert contents[0].startswith('<conversation_summary covered_until_message_id="msg_cpx_4">')
    assert "压缩后的摘要内容" in contents[0]
    # 被覆盖的旧消息不再进历史，覆盖点之后的保留
    assert not any("第1 条历史消息" in c or "第4 条历史消息" in c for c in contents)
    assert any("第5 条历史消息" in c for c in contents)
    assert any("第10 条历史消息" in c for c in contents)


async def test_history_summary_survives_token_budget(monkeypatch):
    """摘要块视同 pinned —— 预算再紧也不能被丢掉。"""
    await _setup()
    await _add_n_messages("msg_cpx_", 10)
    _patch_choice(monkeypatch, _fake_choice())
    async with SessionLocal() as session:
        await compact_conversation(session, CONV)

    async with SessionLocal() as session:
        history = await build_history_for(session, AG, CONV, token_budget=1)

    assert len(history) == 1
    assert history[0]["content"].startswith("<conversation_summary ")


# ─── prompt 前缀（不消费 history 的 adapter 用）─────────────────


async def test_prefix_prompt_with_context_summary(monkeypatch):
    await _setup()
    await _add_n_messages("msg_cpx_", 8)
    _patch_choice(monkeypatch, _fake_choice())
    async with SessionLocal() as session:
        await compact_conversation(session, CONV)

    async with SessionLocal() as session:
        prefixed = await prefix_prompt_with_context_summary(session, CONV, "用户的新指令")

    assert prefixed.startswith('<conversation_summary covered_until_message_id="')
    assert prefixed.endswith("用户的新指令")
    assert "压缩后的摘要内容" in prefixed


async def test_prefix_prompt_without_summary_is_unchanged():
    await _setup()
    async with SessionLocal() as session:
        result = await prefix_prompt_with_context_summary(session, CONV, "用户的新指令")
    assert result == "用户的新指令"


# ─── POST /api/conversations/{id}/compact 契约 ─────────────────


async def test_compact_route_returns_summary_and_message(client, conversation, monkeypatch):
    conv_id = conversation["id"]
    # 直插消息（不走发消息接口）：避免 mock run 的回复消息落库时机影响计数
    for i in range(8):
        _CLOCK[0] += 1
        async with SessionLocal() as session:
            session.add(
                Message(
                    id=new_message_id(),
                    conversation_id=conv_id,
                    role="user",
                    agent_id=None,
                    parts=[{"type": "text", "content": f"消息 {i}"}],
                    status="complete",
                    parent_message_id=None,
                    mentioned_agent_ids=[],
                    run_id=None,
                    usage=None,
                    created_at=_CLOCK[0],
                )
            )
            await session.commit()
    _patch_choice(monkeypatch, _fake_choice())

    res = await client.post(f"/api/conversations/{conv_id}/compact")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["summary"]["id"].startswith("ctx_")
    assert body["summary"]["conversationId"] == conv_id
    assert body["summary"]["coveredUntilMessageId"].startswith("msg_")
    assert body["summary"]["sourceMessageCount"] == 2  # 8 条压掉 2 条
    assert body["summary"]["tokenEstimate"] == estimate_tokens("压缩后的摘要内容")
    assert body["message"]["role"] == "system"
    assert body["message"]["parts"][0]["content"] == "已压缩早期上下文，覆盖 2 条消息。"


async def test_compact_route_unknown_conversation_is_400(client):
    res = await client.post("/api/conversations/conv_nope/compact")
    assert res.status_code == 400
    assert res.json() == {"error": "Conversation not found: conv_nope"}


async def test_compact_route_without_history_is_400(client, conversation):
    res = await client.post(f"/api/conversations/{conversation['id']}/compact")
    assert res.status_code == 400
    assert res.json() == {"error": "No compactable history yet"}
