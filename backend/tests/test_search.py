"""search_messages 的行为测试 + GET /api/search 的查询参数契约。

FTS5 主路径与 LIKE 降级路径各测一遍；查询构造里的 maybeQuote 语义
（前缀 `*` 透传、`(` 让 FTS5 报语法错、`-` 加引号）由这些用例锁住。
"""

from __future__ import annotations

import json
from typing import Any

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.message_search import install_message_search
from app.services.search_service import maybe_quote, search_messages

_SCHEMA_DDL = [
    """CREATE TABLE conversations (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL
    )""",
    """CREATE TABLE agents (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      avatar TEXT NOT NULL
    )""",
    """CREATE TABLE messages (
      id TEXT PRIMARY KEY,
      conversation_id TEXT NOT NULL,
      role TEXT NOT NULL,
      agent_id TEXT,
      parts TEXT NOT NULL,
      status TEXT NOT NULL,
      created_at INTEGER NOT NULL
    )""",
]

_SEED_ROWS = [
    ("INSERT INTO conversations (id, title) VALUES ('c1', 'First conv')"),
    ("INSERT INTO conversations (id, title) VALUES ('c2', 'Second conv')"),
    ("INSERT INTO agents (id, name, avatar) VALUES ('a1', 'Claude', '🤖')"),
]


_open_pairs: list[tuple[AsyncSession, Any]] = []


@pytest_asyncio.fixture(autouse=True)
async def _dispose_open_sessions():
    yield
    while _open_pairs:
        session, engine = _open_pairs.pop()
        await session.close()
        await engine.dispose()


async def make_session(tmp_path) -> AsyncSession:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'search.db'}")
    async with engine.begin() as conn:
        for stmt in _SCHEMA_DDL:
            await conn.exec_driver_sql(stmt)
        for stmt in _SEED_ROWS:
            await conn.exec_driver_sql(stmt)
        await conn.run_sync(install_message_search)
    session = async_sessionmaker(engine, expire_on_commit=False)()
    _open_pairs.append((session, engine))
    return session


async def insert_message(
    session: AsyncSession,
    message_id: str,
    conv_id: str,
    role: str,
    parts: list[dict],
    status: str = "complete",
    agent_id: str | None = None,
) -> None:
    await session.execute(
        text(
            "INSERT INTO messages"
            " (id, conversation_id, role, agent_id, parts, status, created_at)"
            " VALUES (:id, :conv, :role, :agent, :parts, :status, :created)"
        ),
        {
            "id": message_id,
            "conv": conv_id,
            "role": role,
            "agent": agent_id,
            "parts": json.dumps(parts, ensure_ascii=False),
            "status": status,
            "created": 1,
        },
    )
    await session.commit()


# ─── maybeQuote ─────────────────────────────────────────────


def test_maybe_quote_rules():
    assert maybe_quote("render*") == "render*"  # 前缀通配透传
    assert maybe_quote("(unclosed") == "(unclosed"  # 让 FTS5 抛语法错
    assert maybe_quote("a-b") == '"a-b"'  # 连字符会被误读成列限定，加引号
    assert maybe_quote('a"b-c') == '"a""b-c"'  # 引号按 FTS5 规则转义
    assert maybe_quote("plain") == "plain"


# ─── FTS5 主路径 ────────────────────────────────────────────


async def test_returns_empty_result_for_empty_query(tmp_path):
    session = await make_session(tmp_path)
    result = await search_messages(session, query="")
    assert result["hits"] == []
    assert result["total"] == 0


async def test_returns_empty_result_for_whitespace_query(tmp_path):
    session = await make_session(tmp_path)
    result = await search_messages(session, query="   ")
    assert result["hits"] == []
    assert result["total"] == 0


async def test_matches_english_prefix_with_snippet(tmp_path):
    session = await make_session(tmp_path)
    await insert_message(
        session,
        "m1",
        "c1",
        "user",
        [{"type": "text", "content": "rendering pipeline discussion"}],
    )
    result = await search_messages(session, query="render*")
    assert result["total"] == 1
    assert result["hits"][0]["messageId"] == "m1"
    assert "<mark>" in result["hits"][0]["snippetHtml"]


async def test_matches_chinese_substring(tmp_path):
    session = await make_session(tmp_path)
    await insert_message(
        session,
        "m1",
        "c1",
        "user",
        [{"type": "text", "content": "渲染管线优化方案"}],
    )
    result = await search_messages(session, query="渲染管")
    assert result["total"] == 1


async def test_returns_conversation_title_and_agent_name_via_join(tmp_path):
    session = await make_session(tmp_path)
    await insert_message(
        session,
        "m1",
        "c2",
        "agent",
        [{"type": "text", "content": "switching to opus model"}],
        "complete",
        "a1",
    )
    result = await search_messages(session, query="opus")
    assert result["hits"][0]["conversationTitle"] == "Second conv"
    assert result["hits"][0]["agentName"] == "Claude"
    assert result["hits"][0]["agentAvatar"] == "🤖"


async def test_filters_by_conversation_id(tmp_path):
    session = await make_session(tmp_path)
    await insert_message(
        session, "m1", "c1", "user", [{"type": "text", "content": "shared term"}]
    )
    await insert_message(
        session, "m2", "c2", "user", [{"type": "text", "content": "shared term"}]
    )
    result = await search_messages(session, query="shared", conversation_id="c1")
    assert result["total"] == 1
    assert result["hits"][0]["conversationId"] == "c1"


async def test_respects_limit(tmp_path):
    session = await make_session(tmp_path)
    for i in range(5):
        await insert_message(
            session, f"m{i}", "c1", "user", [{"type": "text", "content": "bulk"}]
        )
    result = await search_messages(session, query="bulk", limit=2)
    assert len(result["hits"]) == 2


async def test_returns_error_code_for_invalid_fts5_syntax(tmp_path):
    session = await make_session(tmp_path)
    result = await search_messages(session, query="(unclosed")
    assert result["hits"] == []
    assert result["error"] == "INVALID_QUERY"


async def test_like_fallback_matches_short_chinese_query(tmp_path):
    session = await make_session(tmp_path)
    await insert_message(
        session, "m1", "c1", "user", [{"type": "text", "content": "模型切换问题"}]
    )
    result = await search_messages(session, query="模型", fallback="like")
    assert result["total"] == 1
    # LIKE 路径的片段是纯文本，不带 <mark>
    assert "<mark>" not in result["hits"][0]["snippetHtml"]


async def test_like_fallback_filters_by_conversation_id(tmp_path):
    session = await make_session(tmp_path)
    await insert_message(session, "m1", "c1", "user", [{"type": "text", "content": "共同"}])
    await insert_message(session, "m2", "c2", "user", [{"type": "text", "content": "共同"}])
    result = await search_messages(
        session, query="共同", fallback="like", conversation_id="c2"
    )
    assert result["total"] == 1
    assert result["hits"][0]["conversationId"] == "c2"


# ─── GET /api/search 查询参数契约 ───────────────────────────


async def test_api_returns_400_when_q_is_missing(client):
    res = await client.get("/api/search")
    assert res.status_code == 400
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "INVALID_QUERY"


async def test_api_returns_400_when_q_is_empty_string(client):
    res = await client.get("/api/search?q=")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_QUERY"


async def test_api_returns_400_when_q_is_too_long(client):
    res = await client.get(f"/api/search?q={'x' * 201}")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_QUERY"


async def test_api_returns_400_when_limit_out_of_range(client):
    res = await client.get("/api/search?q=foo&limit=999")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_QUERY"


async def test_api_success_envelope_has_ok_and_data(client):
    res = await client.get("/api/search?q=anything")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["data"]["hits"] == []
    assert body["data"]["total"] == 0
    assert "tookMs" in body["data"]
