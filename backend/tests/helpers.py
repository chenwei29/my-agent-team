"""测试辅助：等 run 收敛、订阅事件总线收事件。

⚠️ 不能只靠「表里没有 running 的 run」来判断收敛 —— POST 返回时后台任务还没被调度，
那一刻表里一条 run 都没有，直接判"已收敛"会拿到空结果。所以要按 runId 等。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sqlalchemy import select

from app.db.models import AgentRun, Message
from app.db.session import SessionLocal
from app.services.event_bus import event_bus

TERMINAL_RUN_STATUSES = ("complete", "failed", "aborted")


async def run_rows(run_ids: list[str]) -> dict[str, AgentRun]:
    if not run_ids:
        return {}
    async with SessionLocal() as session:
        rows = list(await session.scalars(select(AgentRun).where(AgentRun.id.in_(run_ids))))
    return {row.id: row for row in rows}


async def run_row(run_id: str) -> AgentRun | None:
    async with SessionLocal() as session:
        return await session.scalar(select(AgentRun).where(AgentRun.id == run_id))


async def wait_for_runs(run_ids: list[str], timeout: float = 20.0) -> None:
    """等这些 run 全部落终态（complete / failed / aborted）。"""
    if not run_ids:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = await run_rows(run_ids)
        if len(rows) == len(run_ids) and all(
            row.status in TERMINAL_RUN_STATUSES for row in rows.values()
        ):
            return
        await asyncio.sleep(0.02)
    rows = await run_rows(run_ids)
    raise AssertionError(
        f"runs did not settle within {timeout}s: "
        f"{ {rid: (row.status, row.error) for rid, row in rows.items()} }"
    )


async def wait_for_conversation_runs(
    conversation_id: str, min_runs: int = 1, timeout: float = 20.0
) -> None:
    """等某个会话的 run 起来（至少 min_runs 个）并全部落终态。"""
    deadline = time.monotonic() + timeout
    seen = 0
    while time.monotonic() < deadline:
        async with SessionLocal() as session:
            rows = list(
                await session.scalars(
                    select(AgentRun).where(AgentRun.conversation_id == conversation_id)
                )
            )
        seen = max(seen, len(rows))
        if seen >= min_runs and all(row.status in TERMINAL_RUN_STATUSES for row in rows):
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"conversation runs did not settle within {timeout}s (seen={seen})")


async def messages_of(conversation_id: str, role: str | None = None) -> list[Message]:
    async with SessionLocal() as session:
        stmt = select(Message).where(Message.conversation_id == conversation_id)
        if role is not None:
            stmt = stmt.where(Message.role == role)
        return list(await session.scalars(stmt.order_by(Message.created_at.asc())))


class EventCollector:
    """订阅事件总线把事件收进列表（测试用；不影响其它订阅者）。"""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> EventCollector:
        self._task = asyncio.create_task(self._consume())
        await asyncio.sleep(0)  # 让订阅先建立
        return self

    async def _consume(self) -> None:
        async for event in event_bus.subscribe():
            self.events.append(event)

    async def __aexit__(self, *_exc: object) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def types(self) -> list[str]:
        return [e.type for e in self.events]

    def of_type(self, type_name: str) -> list[Any]:
        return [e for e in self.events if e.type == type_name]
