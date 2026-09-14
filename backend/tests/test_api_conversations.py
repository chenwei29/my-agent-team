"""`/api/conversations` 契约测试 —— 重点是 PATCH 一个端点 5 种语义 + 级联删除。"""

from __future__ import annotations

from httpx import AsyncClient

from tests.conftest import MOCK_AGENT_ID

CONV_CAMEL_KEYS = {
    "agentIds",
    "pinnedMessageIds",
    "bookmarkedMessageIds",
    "pinnedAt",
    "fsWriteApprovalMode",
    "createdAt",
    "updatedAt",
    "workspaceMode",
    "workspaceBoundPath",
}


async def test_create_single_conversation(client: AsyncClient):
    res = await client.post(
        "/api/conversations", json={"mode": "single", "agentIds": [MOCK_AGENT_ID]}
    )
    assert res.status_code == 201, res.text
    conv = res.json()["conversation"]

    assert conv["id"].startswith("conv_")
    assert conv["mode"] == "single"
    assert conv["agentIds"] == [MOCK_AGENT_ID]
    assert conv["pinnedMessageIds"] == []
    assert conv["bookmarkedMessageIds"] == []
    assert conv["archived"] is False
    assert conv["pinnedAt"] is None
    assert conv["fsWriteApprovalMode"] == "review"
    # ConversationWithMeta 必须带 workspace 信息
    assert conv["workspaceMode"] == "sandbox"
    assert conv["workspaceBoundPath"] is None
    assert CONV_CAMEL_KEYS <= set(conv)
    assert isinstance(conv["createdAt"], int) and conv["createdAt"] > 10**12
    # 单聊默认标题
    assert conv["title"] == "与 E2E Mock 的对话"

    await client.delete(f"/api/conversations/{conv['id']}")


async def test_create_group_conversation_joins_names(client: AsyncClient):
    res = await client.post(
        "/api/conversations", json={"mode": "group", "agentIds": ["ag_pm", "ag_reviewer"]}
    )
    assert res.status_code == 201
    conv = res.json()["conversation"]
    assert conv["mode"] == "group"
    assert conv["title"] == "PM 小灰 / Reviewer"

    await client.delete(f"/api/conversations/{conv['id']}")


async def test_create_conversation_validates_agent_count(client: AsyncClient):
    res = await client.post(
        "/api/conversations", json={"mode": "single", "agentIds": ["ag_pm", "ag_reviewer"]}
    )
    assert res.status_code == 400
    assert "exactly one agent" in res.json()["error"]

    res = await client.post(
        "/api/conversations", json={"mode": "group", "agentIds": ["ag_pm"]}
    )
    assert res.status_code == 400
    assert "at least two agents" in res.json()["error"]


async def test_create_conversation_rejects_unknown_agent(client: AsyncClient):
    res = await client.post(
        "/api/conversations", json={"mode": "single", "agentIds": ["ag_nope"]}
    )
    assert res.status_code == 400
    assert "Agents not found: ag_nope" in res.json()["error"]


async def test_create_conversation_with_explicit_title(client: AsyncClient):
    res = await client.post(
        "/api/conversations",
        json={"mode": "single", "agentIds": [MOCK_AGENT_ID], "title": "自定义标题"},
    )
    assert res.status_code == 201
    assert res.json()["conversation"]["title"] == "自定义标题"
    await client.delete(f"/api/conversations/{res.json()['conversation']['id']}")


async def test_create_conversation_bound_path_rejects_sensitive_dir(client: AsyncClient):
    res = await client.post(
        "/api/conversations",
        json={"mode": "single", "agentIds": [MOCK_AGENT_ID], "boundPath": "/etc"},
    )
    assert res.status_code == 400


async def test_create_conversation_accepts_local_bound_path(client: AsyncClient, safe_dir):
    res = await client.post(
        "/api/conversations",
        json={"mode": "single", "agentIds": [MOCK_AGENT_ID], "boundPath": safe_dir},
    )
    assert res.status_code == 201, res.text
    conv = res.json()["conversation"]
    assert conv["workspaceMode"] == "local"
    assert conv["workspaceBoundPath"] == safe_dir

    await client.delete(f"/api/conversations/{conv['id']}")


async def test_patch_rename(client: AsyncClient, conversation: dict):
    res = await client.patch(
        f"/api/conversations/{conversation['id']}", json={"title": "重命名了"}
    )
    assert res.status_code == 200, res.text
    assert res.json()["conversation"]["title"] == "重命名了"


async def test_patch_rename_rejects_empty_title(client: AsyncClient, conversation: dict):
    res = await client.patch(f"/api/conversations/{conversation['id']}", json={"title": ""})
    assert res.status_code == 400


async def test_patch_toggle_pin_round_trip(client: AsyncClient, conversation: dict):
    cid = conversation["id"]

    res = await client.patch(f"/api/conversations/{cid}", json={"togglePin": True})
    assert res.status_code == 200
    pinned_at = res.json()["conversation"]["pinnedAt"]
    assert isinstance(pinned_at, int) and pinned_at > 10**12

    res = await client.patch(f"/api/conversations/{cid}", json={"togglePin": True})
    assert res.json()["conversation"]["pinnedAt"] is None


async def test_patch_toggle_archive_does_not_touch_updated_at(
    client: AsyncClient, conversation: dict
):
    cid = conversation["id"]
    before = conversation["updatedAt"]

    res = await client.patch(f"/api/conversations/{cid}", json={"toggleArchive": True})
    assert res.status_code == 200
    body = res.json()["conversation"]
    assert body["archived"] is True
    # 归档是元操作，不应顶到列表前
    assert body["updatedAt"] == before


async def test_patch_add_agents_promotes_to_group(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    res = await client.patch(f"/api/conversations/{cid}", json={"addAgentIds": ["ag_pm"]})
    assert res.status_code == 200
    body = res.json()["conversation"]
    assert body["mode"] == "group"
    assert body["agentIds"] == [MOCK_AGENT_ID, "ag_pm"]

    # 重复添加要幂等去重
    res = await client.patch(f"/api/conversations/{cid}", json={"addAgentIds": ["ag_pm"]})
    assert res.json()["conversation"]["agentIds"] == [MOCK_AGENT_ID, "ag_pm"]


async def test_patch_fs_write_approval_mode(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    res = await client.patch(f"/api/conversations/{cid}", json={"fsWriteApprovalMode": "auto"})
    assert res.status_code == 200
    assert res.json()["conversation"]["fsWriteApprovalMode"] == "auto"


async def test_patch_rejects_toggle_false_and_empty_body(client: AsyncClient, conversation: dict):
    cid = conversation["id"]

    # togglePin 是 z.literal(true)：传 false 是非法请求体，不是「取消置顶」
    res = await client.patch(f"/api/conversations/{cid}", json={"togglePin": False})
    assert res.status_code == 400

    # 至少要有一个字段
    res = await client.patch(f"/api/conversations/{cid}", json={})
    assert res.status_code == 400


async def test_patch_unknown_conversation_is_400(client: AsyncClient):
    res = await client.patch("/api/conversations/conv_nope", json={"title": "x"})
    assert res.status_code == 400


async def test_list_conversations_pinned_first(client: AsyncClient):
    first = (
        await client.post("/api/conversations", json={"mode": "single", "agentIds": [MOCK_AGENT_ID]})
    ).json()["conversation"]
    second = (
        await client.post("/api/conversations", json={"mode": "single", "agentIds": [MOCK_AGENT_ID]})
    ).json()["conversation"]

    # 把先建的置顶，它应排到最前
    await client.patch(f"/api/conversations/{first['id']}", json={"togglePin": True})

    listed = (await client.get("/api/conversations")).json()["conversations"]
    ids = [c["id"] for c in listed]
    assert ids.index(first["id"]) < ids.index(second["id"])

    await client.delete(f"/api/conversations/{first['id']}")
    await client.delete(f"/api/conversations/{second['id']}")


async def test_delete_conversation_cascades(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    await client.post(f"/api/conversations/{cid}/messages", json={"content": "会被级联删掉"})
    assert len((await client.get(f"/api/conversations/{cid}/messages")).json()["messages"]) == 1

    res = await client.delete(f"/api/conversations/{cid}")
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    # 会话消失
    ids = {c["id"] for c in (await client.get("/api/conversations")).json()["conversations"]}
    assert cid not in ids


async def test_delete_unknown_conversation_is_404(client: AsyncClient):
    res = await client.delete("/api/conversations/conv_nope")
    assert res.status_code == 404
