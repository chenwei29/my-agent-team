"""DB 启动期自举：建表 + 首次启动时 seed 内置 agent。

幂等：create_all 不重复建表；seed 前先查 is_builtin 是否存在，已有就整体跳过。

内置 agent 不做升级补齐：本项目不携带历史 DB，builtin_agents.py 里的文案就是当前最新版，
不需要回头把旧行的 systemPrompt / toolNames 覆盖一遍。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.builtin_agents import builtin_agents
from app.db.models import Agent, Base
from app.db.session import SessionLocal, engine


async def ensure_schema() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def ensure_builtin_agents(session: AsyncSession) -> int:
    """已有任意 builtin agent 就跳过；否则一次插入全部。返回插入条数。"""
    existing = await session.execute(select(Agent.id).where(Agent.is_builtin.is_(True)).limit(1))
    if existing.first() is not None:
        return 0

    rows = builtin_agents()
    session.add_all([Agent(**row) for row in rows])
    await session.commit()
    return len(rows)


async def bootstrap_database() -> None:
    await ensure_schema()
    async with SessionLocal() as session:
        await ensure_builtin_agents(session)
