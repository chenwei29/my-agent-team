"""消息全文搜索：FTS5 主路径 + LIKE 降级路径。

FTS5 主路径（默认）：
- 命中片段用 `snippet(messages_fts, 0, '<mark>', '</mark>', '…', 12)` 生成，含 <mark> 高亮；
- 按 `bm25(messages_fts)` 相关性排序（越小越相关）；
- 用户输入是 FTS5 语法的裸串，`maybeQuote` 只对危险片段做最小转义：
  * 结尾 `*` → 原样透传（前缀匹配）
  * 含 `(` → 原样透传（让 FTS5 抛语法错，由调用方拿到 INVALID_QUERY）
  * 含 `-` → 整串加双引号（否则 FTS5 把它读成列限定查询），内部 `"` 转义成 `""`
- FTS5 语法错误（消息里含 fts5 / SQLITE_ERROR）收敛成 `error='INVALID_QUERY'`，不抛给路由。

LIKE 降级路径（`fallback=like`，给 FTS5 不可用或中文短词场景）：
- 在 `messages.parts` 的原始 JSON 文本上做子串匹配，按 created_at 倒序；
- 片段从 parts JSON 里截 80 字符窗口，**纯文本、无 <mark>**（前端据此区分渲染）。

total 口径：FTS 路径只在「本页刚好填满」时才 COUNT 全量，否则 total 就是本页条数；
LIKE 路径没有全量 COUNT，total 恒等于本页条数。
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.time import now_ms

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100

_FTS_SYNTAX_ERROR = re.compile(r"(?:fts5|SQLITE_ERROR)", re.IGNORECASE)


def maybe_quote(query: str) -> str:
    if query.endswith("*"):
        return query
    if "(" in query:
        return query
    if "-" in query:
        return '"' + query.replace('"', '""') + '"'
    return query


async def search_messages(
    session: AsyncSession,
    *,
    query: str,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
    conversation_id: str | None = None,
    role: str | None = None,
    fallback: str | None = None,
) -> dict[str, Any]:
    """返回 {hits, total, tookMs, error?}；error 只有 'INVALID_QUERY' 一种。"""
    trimmed = query.strip()
    if not trimmed:
        return {"hits": [], "total": 0, "tookMs": 0}

    limit = min(max(limit, 1), _MAX_LIMIT)
    offset = max(offset, 0)

    if fallback == "like":
        return await _run_like_path(
            session, trimmed, limit, offset, conversation_id, role
        )
    return await _run_fts_path(
        session, trimmed, limit, offset, conversation_id, role
    )


async def _run_fts_path(
    session: AsyncSession,
    query: str,
    limit: int,
    offset: int,
    conversation_id: str | None,
    role: str | None,
) -> dict[str, Any]:
    start = now_ms()
    fts_query = maybe_quote(query)
    params = {
        "q": fts_query,
        "conv": conversation_id,
        "role": role,
        "limit": limit,
        "offset": offset,
    }
    try:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT
                      m.id AS messageId,
                      m.conversation_id AS conversationId,
                      m.role AS role,
                      m.agent_id AS agentId,
                      m.created_at AS createdAt,
                      snippet(messages_fts, 0, '<mark>', '</mark>', '…', 12) AS snippetHtml,
                      c.title AS conversationTitle,
                      a.name AS agentName,
                      a.avatar AS agentAvatar
                    FROM messages_fts
                    JOIN messages m      ON m.rowid = messages_fts.rowid
                    JOIN conversations c ON c.id = m.conversation_id
                    LEFT JOIN agents a   ON a.id = m.agent_id
                    WHERE messages_fts MATCH :q
                      AND (:conv IS NULL OR m.conversation_id = :conv)
                      AND (:role IS NULL OR m.role = :role)
                    ORDER BY bm25(messages_fts)
                    LIMIT :limit OFFSET :offset
                    """
                ),
                params,
            )
        ).mappings().all()
    except Exception as err:  # noqa: BLE001 - 只把 FTS5 语法错收敛成业务错误
        if _FTS_SYNTAX_ERROR.search(str(err)):
            return {"hits": [], "total": 0, "tookMs": 0, "error": "INVALID_QUERY"}
        raise

    total = len(rows)
    if len(rows) == limit:
        count_row = (
            await session.execute(
                text(
                    """
                    SELECT COUNT(*) AS n FROM messages_fts
                    JOIN messages m ON m.rowid = messages_fts.rowid
                    WHERE messages_fts MATCH :q
                      AND (:conv IS NULL OR m.conversation_id = :conv)
                      AND (:role IS NULL OR m.role = :role)
                    """
                ),
                {
                    "q": fts_query,
                    "conv": conversation_id,
                    "role": role,
                },
            )
        ).scalar_one()
        total = int(count_row)

    return {
        "hits": [dict(r) for r in rows],
        "total": total,
        "tookMs": now_ms() - start,
    }


async def _run_like_path(
    session: AsyncSession,
    query: str,
    limit: int,
    offset: int,
    conversation_id: str | None,
    role: str | None,
) -> dict[str, Any]:
    start = now_ms()
    rows = (
        await session.execute(
            text(
                """
                SELECT
                  m.id AS messageId,
                  m.conversation_id AS conversationId,
                  m.role AS role,
                  m.agent_id AS agentId,
                  m.created_at AS createdAt,
                  substr(m.parts, max(1, instr(m.parts, :q) - 30), 80) AS snippetHtml,
                  c.title AS conversationTitle,
                  a.name AS agentName,
                  a.avatar AS agentAvatar
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                LEFT JOIN agents a   ON a.id = m.agent_id
                WHERE m.parts LIKE '%' || :q || '%'
                  AND (:conv IS NULL OR m.conversation_id = :conv)
                  AND (:role IS NULL OR m.role = :role)
                ORDER BY m.created_at DESC
                LIMIT :limit OFFSET :offset
                """
            ),
            {
                "q": query,
                "conv": conversation_id,
                "role": role,
                "limit": limit,
                "offset": offset,
            },
        )
    ).mappings().all()

    return {
        "hits": [dict(r) for r in rows],
        "total": len(rows),
        "tookMs": now_ms() - start,
    }
