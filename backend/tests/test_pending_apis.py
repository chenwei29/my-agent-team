"""/api/conversations/{id}/pending-writes 端点的状态码矩阵与列表语义。

bash / ask_user 的同款端点用例在各自里程碑补进本文件。
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.services.pending_bash_commands import pending_bash_commands
from app.services.pending_writes import pending_writes
from tests.test_fs_service import _Ws  # 复用 workspace 替身


def _register_pending(conv_id: str, path: str, run_id: str = "run_pw_api") -> dict:
    return pending_writes.register(
        conversation_id=conv_id,
        agent_id="ag_pw_api",
        run_id=run_id,
        path=path,
        absolute_path=f"/tmp/{path}",
        old_content=None,
        new_content="x",
        workspace=_Ws(mode="sandbox", root_path="/tmp"),
    )


@pytest.fixture(autouse=True)
def _clean_store():
    yield
    pending_writes._entries.clear()
    pending_writes._extra.clear()
    pending_bash_commands._entries.clear()


@pytest.mark.asyncio
async def test_list_pending_writes_sorted_by_created_at(client: AsyncClient):
    first = _register_pending("conv_pw_a", "a.txt", run_id="run_pw_1")
    second = _register_pending("conv_pw_a", "b.txt", run_id="run_pw_2")
    _register_pending("conv_pw_b", "c.txt")  # 别的会话，不该出现

    res = await client.get("/api/conversations/conv_pw_a/pending-writes")
    assert res.status_code == 200, res.text
    body = res.json()
    assert [p["id"] for p in body["pendingWrites"]] == [first["id"], second["id"]]
    assert all(p["path"] in ("a.txt", "b.txt") for p in body["pendingWrites"])


@pytest.mark.asyncio
async def test_resolve_invalid_body_is_400(client: AsyncClient):
    pending = _register_pending("conv_pw_a", "a.txt")
    res = await client.post(
        f"/api/conversations/conv_pw_a/pending-writes/{pending['id']}", json={"action": "maybe"}
    )
    assert res.status_code == 400, res.text
    assert res.json()["error"] == "Invalid body"


@pytest.mark.asyncio
async def test_resolve_unknown_id_is_404(client: AsyncClient):
    res = await client.post(
        "/api/conversations/conv_pw_a/pending-writes/pwr_missing", json={"action": "approve"}
    )
    assert res.status_code == 404, res.text
    assert res.json()["error"] == "Pending write not found"


@pytest.mark.asyncio
async def test_reject_via_api(client: AsyncClient):
    pending = _register_pending("conv_pw_a", "a.txt")
    res = await client.post(
        f"/api/conversations/conv_pw_a/pending-writes/{pending['id']}", json={"action": "reject"}
    )
    assert res.status_code == 200, res.text
    assert res.json() == {"ok": True}
    assert pending_writes.list_by_conversation("conv_pw_a") == []


# ─── pending-bash-commands ───────────────────────────────


def _register_pending_command(conv_id: str, command: str = "npm install x") -> dict:
    return pending_bash_commands.register(
        conversation_id=conv_id,
        agent_id="ag_pbc_api",
        run_id="run_pbc_api",
        command=command,
        cwd="/tmp",
        reason="package manager changes dependencies or downloads packages",
    )


@pytest.mark.asyncio
async def test_bash_command_conversation_cross_check(client: AsyncClient):
    pending = _register_pending_command("conv_pbc_a")
    # 路径里的 conversationId 与条目不一致 → 404（这条路由做从属校验）
    res = await client.post(
        f"/api/conversations/conv_other/pending-bash-commands/{pending['id']}",
        json={"action": "approve"},
    )
    assert res.status_code == 404, res.text
    assert res.json()["error"] == "Pending command not found"

    # 一致时正常放行
    res = await client.post(
        f"/api/conversations/conv_pbc_a/pending-bash-commands/{pending['id']}",
        json={"action": "reject"},
    )
    assert res.status_code == 200, res.text
    assert pending_bash_commands.list_by_conversation("conv_pbc_a") == []


@pytest.mark.asyncio
async def test_bash_command_list_and_invalid_body(client: AsyncClient):
    first = _register_pending_command("conv_pbc_a", "git reset --hard")
    res = await client.get("/api/conversations/conv_pbc_a/pending-bash-commands")
    assert res.status_code == 200, res.text
    assert [p["id"] for p in res.json()["pendingCommands"]] == [first["id"]]
    assert first["reason"]

    res = await client.post(
        f"/api/conversations/conv_pbc_a/pending-bash-commands/{first['id']}", json={}
    )
    assert res.status_code == 400, res.text
