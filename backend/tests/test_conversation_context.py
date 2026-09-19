"""跨 run 历史序列化（build_history_for）。

直接往临时库插消息行再调纯读函数，验证：user/agent 视角、thinking 与工具
part 不回放、artifact_ref 折叠、pinned 永不截断、token 预算从老往新丢。
"""

from __future__ import annotations

from typing import Any

import pytest_asyncio
from sqlalchemy import delete, select

from app.db.bootstrap import bootstrap_database
from app.db.models import Agent, Artifact, Conversation, Message
from app.db.session import SessionLocal
from app.services.conversation_context import build_history_for
from app.utils.time import now_ms

AG_A = "ag_ctx_a"
AG_B = "ag_ctx_b"
CONV = "conv_ctx"

# 消息时间戳逐条 +1，保证排序确定（同一毫秒内插入的多条消息顺序不稳定）
_CLOCK = [now_ms()]


@pytest_asyncio.fixture(autouse=True)
async def _fresh_db():
    """建表 + 清掉本模块固定 id 的旧数据（tempdir 是整个 session 共享的）。"""
    await bootstrap_database()
    async with SessionLocal() as session:
        await session.execute(delete(Message).where(Message.conversation_id == CONV))
        await session.execute(delete(Artifact).where(Artifact.conversation_id == CONV))
        await session.execute(delete(Conversation).where(Conversation.id == CONV))
        await session.execute(delete(Agent).where(Agent.id.in_([AG_A, AG_B])))
        await session.commit()
    yield


async def _setup(agent_ids: list[str]) -> str:
    async with SessionLocal() as session:
        for agent_id, name in ((AG_A, "Agent A"), (AG_B, "Agent B")):
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
        session.add(
            Conversation(
                id=CONV,
                title="ctx 测试会话",
                mode="group" if len(agent_ids) > 1 else "single",
                agent_ids=agent_ids,
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
    return CONV


async def _add_message(
    msg_id: str,
    role: str,
    parts: list[dict[str, Any]],
    *,
    agent_id: str | None = None,
    status: str = "complete",
) -> str:
    _CLOCK[0] += 1
    async with SessionLocal() as session:
        session.add(
            Message(
                id=msg_id,
                conversation_id=CONV,
                role=role,
                agent_id=agent_id,
                parts=parts,
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


async def _pin(msg_ids: list[str]) -> None:
    async with SessionLocal() as session:
        conv = await session.scalar(select(Conversation).where(Conversation.id == CONV))
        conv.pinned_message_ids = list(msg_ids)
        await session.commit()


async def _history(agent_id: str, **kwargs: Any) -> list[dict[str, Any]]:
    async with SessionLocal() as session:
        return await build_history_for(session, agent_id, CONV, **kwargs)


# ─── 单聊（A 是当前 agent） ───────────────────────────────


async def test_user_message_and_self_assistant():
    await _setup([AG_A])
    await _add_message("msg_1", "user", [{"type": "text", "content": "你好"}])
    await _add_message(
        "msg_2",
        "agent",
        [
            {"type": "thinking", "content": "内心戏"},
            {"type": "text", "content": "你好，有什么可以帮你？"},
        ],
        agent_id=AG_A,
    )

    history = await _history(AG_A)
    assert history == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]


async def test_tool_parts_and_attachments_placeholder():
    await _setup([AG_A])
    await _add_message(
        "msg_1",
        "user",
        [
            {"type": "text", "content": "看图"},
            {"type": "image_attachment", "attachmentId": "att_1", "fileName": "cat.png", "size": 1, "mimeType": "image/png"},
            {"type": "file_attachment", "attachmentId": "att_2", "fileName": "a.txt", "size": 1, "mimeType": "text/plain"},
        ],
    )
    await _add_message(
        "msg_2",
        "agent",
        [
            {"type": "tool_use", "callId": "call_1", "toolName": "bash", "args": {}},
            {"type": "tool_result", "callId": "call_1", "result": "ok", "isError": False},
            {"type": "text", "content": "看完了"},
        ],
        agent_id=AG_A,
    )

    history = await _history(AG_A)
    assert history[0]["content"] == "看图\n[图片附件: cat.png]\n[文件附件: a.txt]"
    # tool_use / tool_result 不回放，只剩公开文本
    assert history[1] == {"role": "assistant", "content": "看完了"}


async def test_streaming_and_error_messages_excluded():
    await _setup([AG_A])
    await _add_message("msg_s", "user", [{"type": "text", "content": "流式中"}], status="streaming")
    await _add_message("msg_e", "user", [{"type": "text", "content": "出错了"}], status="error")
    await _add_message("msg_ok", "user", [{"type": "text", "content": "正常"}])

    history = await _history(AG_A)
    assert history == [{"role": "user", "content": "正常"}]


async def test_empty_public_text_agent_message_skipped():
    await _setup([AG_A])
    await _add_message(
        "msg_1", "agent", [{"type": "tool_use", "callId": "c", "toolName": "bash", "args": {}}],
        agent_id=AG_A,
    )

    assert await _history(AG_A) == []


async def test_artifact_ref_folding():
    await _setup([AG_A])
    async with SessionLocal() as session:
        session.add(
            Artifact(
                id="art_1",
                conversation_id=CONV,
                type="document",
                title="需求文档",
                content={},
                version=1,
                parent_artifact_id=None,
                created_by_agent_id=AG_A,
                created_at=now_ms(),
            )
        )
        await session.commit()

    await _add_message(
        "msg_1",
        "agent",
        [
            {"type": "artifact_ref", "artifactId": "art_1"},
            {"type": "artifact_ref", "artifactId": "art_missing"},
        ],
        agent_id=AG_A,
    )

    history = await _history(AG_A)
    assert history[0]["content"] == "[产物: 需求文档 (id=art_1)]\n[产物 art_missing]"


async def test_exclude_trigger_message():
    await _setup([AG_A])
    await _add_message("msg_1", "user", [{"type": "text", "content": "旧消息"}])
    await _add_message("msg_2", "user", [{"type": "text", "content": "触发消息"}])

    history = await _history(AG_A, exclude_message_id="msg_2")
    assert [m["content"] for m in history] == ["旧消息"]


# ─── 群聊（别的 agent 视角） ───────────────────────────────


async def test_group_chat_other_agent_prefixed_as_user():
    await _setup([AG_A, AG_B])
    await _add_message("msg_1", "user", [{"type": "text", "content": "大家怎么看"}])
    await _add_message(
        "msg_2",
        "agent",
        [
            {"type": "thinking", "content": "内心戏"},
            {"type": "text", "content": "我觉得可行"},
        ],
        agent_id=AG_A,
    )
    await _add_message("msg_3", "agent", [{"type": "text", "content": "我也觉得"}], agent_id=AG_B)

    # B 的视角：A 的发言是 [Agent A] 前缀的 user 消息，自己的是 assistant
    history_b = await _history(AG_B)
    assert history_b == [
        {"role": "user", "content": "大家怎么看"},
        {"role": "user", "content": "[Agent A] 我觉得可行"},
        {"role": "assistant", "content": "我也觉得"},
    ]

    # A 的视角对称
    history_a = await _history(AG_A)
    assert history_a[1] == {"role": "assistant", "content": "我觉得可行"}
    assert history_a[2] == {"role": "user", "content": "[Agent B] 我也觉得"}


async def test_single_chat_other_agent_message_dropped():
    # 单聊（agent_ids 只有 A）里不该出现别的 agent 的消息，直接跳过
    await _setup([AG_A])
    await _add_message("msg_1", "agent", [{"type": "text", "content": "别人的话"}], agent_id=AG_B)

    assert await _history(AG_A) == []


# ─── 截断与预算 ──────────────────────────────────────────


async def test_max_turns_takes_most_recent():
    await _setup([AG_A])
    for i in range(5):
        await _add_message(f"msg_{i}", "user", [{"type": "text", "content": f"消息{i}"}])

    history = await _history(AG_A, max_turns=2)
    assert [m["content"] for m in history] == ["消息3", "消息4"]


async def test_pinned_survives_max_turns():
    await _setup([AG_A])
    for i in range(5):
        await _add_message(f"msg_{i}", "user", [{"type": "text", "content": f"消息{i}"}])
    await _pin(["msg_0"])

    history = await _history(AG_A, max_turns=2)
    assert [m["content"] for m in history] == ["消息0", "消息3", "消息4"]


async def test_pinned_survives_tiny_budget():
    await _setup([AG_A])
    await _add_message("msg_old", "user", [{"type": "text", "content": "很老很老很老很老很老很老的消息"}])
    await _add_message("msg_new", "user", [{"type": "text", "content": "新"}])
    await _pin(["msg_old"])

    # 预算极小：非 pinned 的「新」被丢，pinned 的老消息仍然保留
    history = await _history(AG_A, token_budget=1)
    contents = [m["content"] for m in history]
    assert "很老很老很老很老很老很老的消息" in contents
    assert "新" not in contents


async def test_budget_drops_oldest_first():
    await _setup([AG_A])
    await _add_message("m1", "user", [{"type": "text", "content": "aaaa"}])
    await _add_message("m2", "user", [{"type": "text", "content": "bbbb"}])
    await _add_message("m3", "user", [{"type": "text", "content": "cccc"}])

    # 每条 4/4+4=5 token（4 字符 + 4 metadata）；预算 12 装不下三条，最老的先丢
    history = await _history(AG_A, token_budget=12)
    assert [m["content"] for m in history] == ["bbbb", "cccc"]


async def test_no_budget_means_no_trimming():
    await _setup([AG_A])
    await _add_message("m1", "user", [{"type": "text", "content": "a" * 4000}])
    await _add_message("m2", "user", [{"type": "text", "content": "b" * 4000}])

    history = await _history(AG_A)
    assert len(history) == 2
