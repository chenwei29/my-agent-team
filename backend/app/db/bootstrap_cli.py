"""`python -m app.db.bootstrap_cli` —— 建表 + seed，供 E2E 的后端启动命令调用（web/playwright.config.ts）。

`--insert-mock-agent` 额外插入 E2E 专用 mock agent —— 它只能直接写 DB，因为创建
agent 的 API 禁止 adapter_name='mock'（请求体校验会拦下这种入参，这里同样保持该约束）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import select

from app.config import get_settings
from app.db.bootstrap import bootstrap_database
from app.db.models import Agent
from app.db.session import SessionLocal
from app.utils.time import now_ms

E2E_MOCK_AGENT_ID = "ag_e2e_mock"


async def insert_mock_agent() -> None:
    async with SessionLocal() as session:
        existing = await session.execute(
            select(Agent.id).where(Agent.id == E2E_MOCK_AGENT_ID)
        )
        if existing.first() is not None:
            print(f"[bootstrap] skip {E2E_MOCK_AGENT_ID} (already exists)")
            return
        session.add(
            Agent(
                id=E2E_MOCK_AGENT_ID,
                name="E2E Mock",
                avatar="🤖",
                description="E2E 测试专用 mock agent（确定性脚本回复，不调真实 LLM）",
                capabilities=["test"],
                system_prompt="mock",
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
        await session.commit()
        print(f"[bootstrap] insert {E2E_MOCK_AGENT_ID} (E2E Mock)")


async def _main(with_mock_agent: bool) -> None:
    settings = get_settings()
    settings.ensure_dirs()
    print(f"[bootstrap] data dir: {settings.data_dir}")
    await bootstrap_database()
    if with_mock_agent:
        await insert_mock_agent()
    print("[bootstrap] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the AgentHub SQLite database.")
    parser.add_argument(
        "--insert-mock-agent",
        action="store_true",
        help="同时插入 E2E 专用 mock agent（id=ag_e2e_mock）",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_main(args.insert_mock_agent))
    except Exception as err:  # noqa: BLE001 - CLI 顶层，打印后退出
        print(f"[bootstrap] failed: {err}", file=sys.stderr)
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
