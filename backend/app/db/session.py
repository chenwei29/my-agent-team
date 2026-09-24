"""async engine + session factory。

SQLite 外键默认关闭 —— 不开 PRAGMA foreign_keys=ON 的话 ON DELETE CASCADE 完全不生效，
删会话时 messages / workspaces 会变成孤儿行。

WAL + busy_timeout：P2 起多个 run 任务会并发写同一个库文件（每个 run 一个 session），
默认的 rollback journal 模式下并发写会直接 `database is locked`。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

_settings = get_settings()

# JSON 列按「字面 Unicode」落盘（中文不写成 \uXXXX 转义）——搜索的 LIKE 降级路径
# 要在 messages.parts 的原始 JSON 文本上做子串匹配，转义存储会让中文查询打不中。
# 读回不受影响（两种编码都能 loads）。
engine = create_async_engine(
    _settings.database_url,
    future=True,
    json_serializer=lambda v: json.dumps(v, ensure_ascii=False),
)

# 并发 run 抢写时等锁的上限（毫秒）；超过就报错，而不是立刻 database is locked
BUSY_TIMEOUT_MS = 5000


@event.listens_for(engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    cursor.close()


SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：一个请求一个 session。"""
    async with SessionLocal() as session:
        yield session
