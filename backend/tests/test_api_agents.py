"""`/api/agents` 契约测试。"""

from __future__ import annotations

from httpx import AsyncClient

from tests.conftest import MOCK_AGENT_ID

BUILTIN_IDS = {"ag_orchestrator", "ag_pm", "ag_designer", "ag_frontend", "ag_reviewer"}

AGENT_CAMEL_KEYS = {
    "systemPrompt",
    "adapterName",
    "modelProvider",
    "modelId",
    "apiKey",
    "apiBaseUrl",
    "toolNames",
    "isBuiltin",
    "isOrchestrator",
    "supportsVision",
    "createdAt",
}


async def test_list_agents_returns_builtins_in_camel_case(client: AsyncClient):
    res = await client.get("/api/agents")
    assert res.status_code == 200
    agents = res.json()["agents"]

    ids = {a["id"] for a in agents}
    assert BUILTIN_IDS <= ids
    assert MOCK_AGENT_ID in ids

    for agent in agents:
        assert AGENT_CAMEL_KEYS <= set(agent)
        # 时间戳必须是毫秒整数，不是 ISO 字符串
        assert isinstance(agent["createdAt"], int)
        assert agent["createdAt"] > 10**12


async def test_list_agents_builtins_first(client: AsyncClient):
    agents = (await client.get("/api/agents")).json()["agents"]
    flags = [a["isBuiltin"] for a in agents]
    # 内置在前：一旦出现 False，后面就不应再有 True
    assert flags == sorted(flags, reverse=True)


async def test_create_custom_agent(client: AsyncClient):
    res = await client.post(
        "/api/agents",
        json={
            "name": "测试 Agent",
            "description": "用于单测的自建 agent",
            "systemPrompt": "你是测试 agent",
            "modelProvider": "deepseek",
            "modelId": "deepseek-v4-flash",
            "capabilities": ["test"],
            "toolNames": ["fs_read"],
        },
    )
    assert res.status_code == 201, res.text
    agent = res.json()["agent"]
    assert agent["id"].startswith("ag_")
    assert agent["name"] == "测试 Agent"
    assert agent["isBuiltin"] is False
    assert agent["adapterName"] == "custom"
    assert agent["modelProvider"] == "deepseek"
    # avatar 未传时补默认 🤖
    assert agent["avatar"] == "🤖"

    await client.delete(f"/api/agents/{agent['id']}")


async def test_create_custom_agent_requires_provider_and_model(client: AsyncClient):
    res = await client.post(
        "/api/agents",
        json={
            "name": "缺 model 的 agent",
            "description": "应当被拒",
            "systemPrompt": "x",
            "adapterName": "custom",
        },
    )
    assert res.status_code == 400
    assert res.json()["error"] == "Invalid body"


async def test_create_agent_rejects_mock_adapter(client: AsyncClient):
    """adapter_name='mock' 不允许由 API 写入（只有 bootstrap_cli 能直接落库）。"""
    res = await client.post(
        "/api/agents",
        json={
            "name": "mock",
            "description": "非法 adapter",
            "systemPrompt": "x",
            "adapterName": "mock",
        },
    )
    assert res.status_code == 400


async def test_openai_compatible_requires_base_url_and_key(client: AsyncClient):
    res = await client.post(
        "/api/agents",
        json={
            "name": "compat",
            "description": "缺 base url",
            "systemPrompt": "x",
            "modelProvider": "openai-compatible",
            "modelId": "some-model",
        },
    )
    assert res.status_code == 400
    assert "Base URL" in res.json()["error"]


async def test_update_agent_and_reject_unknown_field(client: AsyncClient):
    created = (
        await client.post(
            "/api/agents",
            json={
                "name": "待改",
                "description": "原始描述",
                "systemPrompt": "原始 prompt",
                "modelProvider": "deepseek",
                "modelId": "deepseek-v4-flash",
            },
        )
    ).json()["agent"]
    agent_id = created["id"]

    res = await client.patch(f"/api/agents/{agent_id}", json={"name": "改过名字"})
    assert res.status_code == 200, res.text
    assert res.json()["agent"]["name"] == "改过名字"

    # 请求体按严格模式校验：多余字段直接 400
    res = await client.patch(f"/api/agents/{agent_id}", json={"nope": 1})
    assert res.status_code == 400

    await client.delete(f"/api/agents/{agent_id}")


async def test_delete_builtin_agent_is_forbidden(client: AsyncClient):
    res = await client.delete("/api/agents/ag_pm")
    assert res.status_code == 400
    assert "Built-in agents cannot be deleted" in res.json()["error"]

    # 还在
    agents = (await client.get("/api/agents")).json()["agents"]
    assert "ag_pm" in {a["id"] for a in agents}


async def test_delete_unknown_agent_is_400_not_404(client: AsyncClient):
    """DELETE 出错一律返回 400（不是 404）。"""
    res = await client.delete("/api/agents/ag_does_not_exist")
    assert res.status_code == 400


async def test_agent_draft_requires_intent(client: AsyncClient):
    """草稿生成收下 intent 才能干活；空请求体按 Invalid body 拒绝（行为测试见 test_agent_draft.py）。"""
    res = await client.post("/api/agents/draft", json={})
    assert res.status_code == 400
