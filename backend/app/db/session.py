"""async engine + session factory。

SQLite 外键默认关闭 —— 不开 PRAGMA foreign_keys=ON 的话 ON DELETE CASCADE 完全不生效，
删会话时 messages / workspaces 会变成孤儿行。

WAL + busy_timeout：P2 起多个 run 任务会并发写同一个库文件（每个 run 一个 session），
默认的 rollback journal 模式下并发写会直接 `database is locked`。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

_settings = get_settings()

engine = create_async_engine(_settings.database_url, future=True)

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
