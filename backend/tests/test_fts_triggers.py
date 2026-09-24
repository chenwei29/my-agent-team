"""messages_fts 触发器与索引安装的行为测试（纯 SQL，直接跑真 SQLite）。

验证几条反直觉语义：
- 只有 text part 入索引，thinking 不入；
- streaming 状态的消息不入索引，等 status 变更后才首次入；
- 删除消息会同步删掉索引行；
- 安装过程幂等，且能回填存量消息。
"""

from __future__ import annotations

import json
import sqlite3

from app.db.message_search import install_message_search

_MESSAGES_DDL = """
CREATE TABLE messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  role TEXT NOT NULL,
  agent_id TEXT,
  parts TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
"""


def make_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(_MESSAGES_DDL)
    install_message_search(db)
    return db


def insert_message(
    db: sqlite3.Connection,
    message_id: str,
    parts: list[dict],
    status: str,
) -> None:
    db.execute(
        "INSERT INTO messages (id, conversation_id, role, parts, status, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (message_id, "c1", "user", json.dumps(parts), status, 1),
    )


def fts_count(db: sqlite3.Connection, content: str) -> int:
    row = db.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
        (f'"{content}"',),
    ).fetchone()
    return int(row[0])


# ─── 触发器行为 ──────────────────────────────────────────────


def test_inserts_one_fts_row_per_text_part():
    db = make_db()
    insert_message(
        db,
        "m1",
        [
            {"type": "text", "content": "hello"},
            {"type": "text", "content": "world"},
            {"type": "thinking", "content": "private"},
        ],
        "complete",
    )
    assert fts_count(db, "hello") == 1
    assert fts_count(db, "world") == 1
    assert fts_count(db, "private") == 0


def test_skips_streaming_messages_on_insert():
    db = make_db()
    insert_message(db, "m1", [{"type": "text", "content": "mid-stream"}], "streaming")
    assert fts_count(db, "mid-stream") == 0


def test_syncs_when_status_transitions_streaming_to_complete():
    db = make_db()
    insert_message(db, "m1", [{"type": "text", "content": "growing"}], "streaming")
    assert fts_count(db, "growing") == 0

    db.execute(
        "UPDATE messages SET status = ?, parts = ? WHERE id = ?",
        ("complete", json.dumps([{"type": "text", "content": "grown text"}]), "m1"),
    )

    assert fts_count(db, "grown text") == 1
    assert fts_count(db, "growing") == 0


def test_skips_update_while_still_streaming():
    db = make_db()
    insert_message(db, "m1", [{"type": "text", "content": "first"}], "streaming")
    db.execute(
        "UPDATE messages SET parts = ? WHERE id = ?",
        (json.dumps([{"type": "text", "content": "second"}]), "m1"),
    )
    assert fts_count(db, "first") == 0
    assert fts_count(db, "second") == 0


def test_removes_fts_rows_on_delete():
    db = make_db()
    insert_message(db, "m1", [{"type": "text", "content": "todelete"}], "complete")
    assert fts_count(db, "todelete") == 1
    db.execute("DELETE FROM messages WHERE id = ?", ("m1",))
    assert fts_count(db, "todelete") == 0


# ─── 索引安装（建表 / 触发器 / 幂等 / 回填）──────────────────


def test_creates_messages_fts_virtual_table():
    db = make_db()
    row = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
    ).fetchone()
    assert row is not None


def test_creates_three_triggers():
    db = make_db()
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'messages_fts_%'"
    ).fetchall()
    names = sorted(r[0] for r in rows)
    assert names == ["messages_fts_ad", "messages_fts_ai", "messages_fts_au"]


def test_install_is_idempotent():
    db = make_db()
    install_message_search(db)  # 再跑一遍不应抛错


def test_backfills_existing_text_parts_from_messages():
    db = sqlite3.connect(":memory:")
    db.executescript(_MESSAGES_DDL)
    db.execute(
        "INSERT INTO messages (id, conversation_id, role, parts, status, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            "m1",
            "c1",
            "user",
            json.dumps(
                [
                    {"type": "text", "content": "hello world"},
                    {"type": "thinking", "content": "internal note"},
                    {"type": "text", "content": "goodbye"},
                ]
            ),
            "complete",
            1,
        ),
    )

    install_message_search(db)

    rows = db.execute("SELECT content FROM messages_fts ORDER BY rowid").fetchall()
    # thinking 被排除；多个 text part 拼进同一行
    assert len(rows) == 1
    assert "hello world" in rows[0][0]
    assert "goodbye" in rows[0][0]
    assert "internal note" not in rows[0][0]
