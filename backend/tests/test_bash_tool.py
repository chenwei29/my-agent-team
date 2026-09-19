"""bash 工具：黑名单拦截、cwd 约束、超时杀进程组、输出截断、审批挂起。

直接调 handler（不经过 LLM），workspace 用真实会话（client 夹具建会话时
自动生成 sandbox workspace 行）。
"""

from __future__ import annotations

import asyncio
import subprocess

import pytest
from httpx import AsyncClient

from app.tools.bash import BASH_TOOL
from app.tools.types import ToolContext
from app.utils.abort import AbortSignal
from app.services.pending_bash_commands import pending_bash_commands
from sqlalchemy import select

from app.db.models import Workspace
from app.db.session import SessionLocal


async def _ctx(client: AsyncClient) -> ToolContext:
    res = await client.post(
        "/api/conversations", json={"mode": "single", "agentIds": ["ag_e2e_mock"]}
    )
    assert res.status_code == 201, res.text
    conv_id = res.json()["conversation"]["id"]
    async with SessionLocal() as session:
        ws = await session.scalar(select(Workspace).where(Workspace.conversation_id == conv_id))
    assert ws is not None
    return ToolContext(
        conversation_id=conv_id,
        agent_id="ag_bash_test",
        run_id="run_bash_test",
        workspace_path=ws.root_path,
        abort_signal=AbortSignal(),
    )


def _no_lingering_sleep() -> bool:
    result = subprocess.run(["pgrep", "-f", "sleep 60"], capture_output=True, text=True)
    return result.returncode != 0


@pytest.fixture(autouse=True)
def _clean_store():
    yield
    pending_bash_commands._entries.clear()


async def test_banned_command_rejected(client: AsyncClient):
    ctx = await _ctx(client)
    result = await BASH_TOOL.handler({"command": "sudo whoami"}, ctx)
    assert result.ok is False
    assert result.error.startswith("Command rejected by safety policy: ")


async def test_echo_runs_and_captures_output(client: AsyncClient):
    ctx = await _ctx(client)
    result = await BASH_TOOL.handler({"command": "echo hello"}, ctx)
    assert result.ok is True, result.error
    assert result.value["exitCode"] == 0
    assert "hello" in result.value["output"]
    assert result.value["timedOut"] is False
    assert result.value["cwd"] == ctx.workspace_path


async def test_nonzero_exit_is_still_ok(client: AsyncClient):
    ctx = await _ctx(client)
    result = await BASH_TOOL.handler({"command": "exit 3"}, ctx)
    assert result.ok is True
    assert result.value["exitCode"] == 3


async def test_cwd_escape_rejected(client: AsyncClient):
    ctx = await _ctx(client)
    result = await BASH_TOOL.handler({"command": "pwd", "cwd": ".."}, ctx)
    assert result.ok is False
    assert "outside workspace" in result.error


async def test_cwd_must_be_directory(client: AsyncClient):
    ctx = await _ctx(client)
    result = await BASH_TOOL.handler({"command": "pwd", "cwd": "no-such-dir"}, ctx)
    assert result.ok is False
    assert result.error == "cwd is not a directory: no-such-dir"


async def test_timeout_kills_process(client: AsyncClient):
    ctx = await _ctx(client)
    result = await BASH_TOOL.handler({"command": "sleep 60", "timeoutMs": 1000}, ctx)
    assert result.ok is True
    assert result.value["timedOut"] is True
    assert "[KILLED after 1.0s timeout]" in result.value["output"]
    # 进程组被杀干净，无残留
    await asyncio.sleep(0.2)
    assert _no_lingering_sleep(), "sleep 进程残留了"


async def test_output_truncated_at_10000_chars(client: AsyncClient):
    ctx = await _ctx(client)
    result = await BASH_TOOL.handler({"command": "printf 'x%.0s' {1..12000}"}, ctx)
    assert result.ok is True
    assert result.value["truncated"] is True
    assert result.value["output"].endswith("\n\n[TRUNCATED at 10000 chars]")


async def test_high_risk_command_requires_approval_then_rejected(client: AsyncClient):
    ctx = await _ctx(client)
    handler_task = asyncio.create_task(
        BASH_TOOL.handler({"command": "npm install left-pad"}, ctx)
    )
    deadline = asyncio.get_event_loop().time() + 5
    pending = None
    while asyncio.get_event_loop().time() < deadline:
        items = pending_bash_commands.list_by_conversation(ctx.conversation_id)
        if items:
            pending = items[0]
            break
        await asyncio.sleep(0.02)
    assert pending is not None
    assert pending["command"] == "npm install left-pad"
    assert pending["reason"]

    pending_bash_commands.reject(pending["id"])
    result = await handler_task
    assert result.ok is False
    assert result.error == "User rejected command execution: package manager changes dependencies or downloads packages"
    assert pending_bash_commands.list_by_conversation(ctx.conversation_id) == []


async def test_high_risk_command_approved_then_runs(client: AsyncClient):
    ctx = await _ctx(client)
    # chmod 命中审批规则，批准后真的执行（无害）
    handler_task = asyncio.create_task(BASH_TOOL.handler({"command": "chmod +x ."}, ctx))
    deadline = asyncio.get_event_loop().time() + 5
    pending = None
    while asyncio.get_event_loop().time() < deadline:
        items = pending_bash_commands.list_by_conversation(ctx.conversation_id)
        if items:
            pending = items[0]
            break
        await asyncio.sleep(0.02)
    assert pending is not None

    pending_bash_commands.approve(pending["id"])
    result = await handler_task
    assert result.ok is True, result.error
    assert result.value["exitCode"] == 0


async def test_abort_mid_command_kills_process(client: AsyncClient):
    ctx = await _ctx(client)
    handler_task = asyncio.create_task(BASH_TOOL.handler({"command": "sleep 60"}, ctx))
    await asyncio.sleep(0.5)
    ctx.abort_signal.abort()
    result = await handler_task
    assert result.ok is True
    assert "[KILLED after run abort]" in result.value["output"]
    await asyncio.sleep(0.2)
    assert _no_lingering_sleep(), "abort 后 sleep 进程残留了"
