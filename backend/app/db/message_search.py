"""消息全文检索索引：`messages_fts` 虚表 + 与 `messages` 同步的三个触发器 + 存量回填。

设计要点（改这里前先弄明白这几条行为语义）：
- 虚表用 `tokenize='trigram'`：中英子串都能搜（中文需 3 字以上），并支持 `render*` 这类前缀匹配。
- 每个 **text part** 贡献一段文本，多个 part 用空格拼接进**同一行**（按 `messages.rowid` 对齐）；
  thinking 等非 text part 不入索引。
- **streaming 状态的消息不入索引**（INSERT 被 WHEN 拦下，UPDATE 期间也不重排）；
  等它变成 complete 的那次 UPDATE 才首次入索引 —— 搜索不该搜到半截内容。
- DELETE 触发器把整行索引删掉，避免消息删除后还能搜到。
- 回填用 `INSERT OR IGNORE`，整个安装过程幂等（IF NOT EXISTS 守卫），每次启动跑一遍无副作用。

`messages` 的 rowid 是 SQLite 隐式主键（`id TEXT PRIMARY KEY` 不占 rowid），
不要改成 INTEGER PRIMARY KEY，否则 rowid 与 id 会互相抢位。
"""

from __future__ import annotations

from typing import Any

MESSAGE_SEARCH_STATEMENTS: list[str] = [
    "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, tokenize='trigram')",
    """CREATE TRIGGER IF NOT EXISTS messages_fts_ai
       AFTER INSERT ON messages
       WHEN new.status != 'streaming'
       BEGIN
         INSERT INTO messages_fts(rowid, content)
         SELECT new.rowid, (
           SELECT GROUP_CONCAT(json_extract(value, '$.content'), ' ')
           FROM json_each(new.parts)
           WHERE json_extract(value, '$.type') = 'text'
         );
       END""",
    """CREATE TRIGGER IF NOT EXISTS messages_fts_au
       AFTER UPDATE ON messages
       WHEN new.status != 'streaming'
       BEGIN
         DELETE FROM messages_fts WHERE rowid = old.rowid;
         INSERT INTO messages_fts(rowid, content)
         SELECT new.rowid, (
           SELECT GROUP_CONCAT(json_extract(value, '$.content'), ' ')
           FROM json_each(new.parts)
           WHERE json_extract(value, '$.type') = 'text'
         );
       END""",
    """CREATE TRIGGER IF NOT EXISTS messages_fts_ad
       AFTER DELETE ON messages
       BEGIN
         DELETE FROM messages_fts WHERE rowid = old.rowid;
       END""",
    """INSERT OR IGNORE INTO messages_fts(rowid, content)
       SELECT m.rowid, (
         SELECT GROUP_CONCAT(json_extract(value, '$.content'), ' ')
         FROM json_each(m.parts)
         WHERE json_extract(value, '$.type') = 'text'
       )
       FROM messages m""",
]


def install_message_search(connection: Any) -> None:
    """在给定连接上安装全文索引（幂等）。连接需支持 exec_driver_sql 或 execute。"""
    for statement in MESSAGE_SEARCH_STATEMENTS:
        if hasattr(connection, "exec_driver_sql"):
            connection.exec_driver_sql(statement)
        else:
            connection.execute(statement)
