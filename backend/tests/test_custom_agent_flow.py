"""custom agent 全链路（HTTP → runner → adapter → 落库）：用假 OpenAI 客户端验证接线。

覆盖：key 三层解析（per-agent / 全局设置 / 环境变量）、system prompt 拼装、
历史注入、流式 part 落库、usage 落到 message 与 run。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.adapters import custom as custom_module
from app.db.models import Agent
from app.db.session import SessionLocal
from tests.helpers import messages_of, run_row, wait_for_runs

AG_CUSTOM = "ag_custom_test"


class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = iter(chunks)

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _chunk(content: str | None = None, finish_reason: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None
    )


def _usage_chunk(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[], usage=usage)


class _Capture:
    """记录 build_client 收到的参数与最终请求的 messages。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.client_args: list[tuple] = []
        self.request_messages: list[list[dict]] = []

    def build(self, provider, override_key, api_base_url):
        self.client_args.append((provider, override_key, api_base_url))
        capture = self

        class _Completions:
            async def create(self, **kwargs):
                capture.request_messages.append(kwargs["messages"])
                reply = capture.replies.pop(0)
                return _FakeStream(
                    [
                        _chunk(content=reply),
                        _chunk(finish_reason="stop"),
                        _usage_chunk(11, 7),
                    ]
                )

        return SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))


async def _ensure_custom_agent(
    api_key: str | None = None, system_prompt: str = "你是测试 agent"
) -> None:
    """upsert：agent 可能已被前一个用例的 run 引用（消息外键），不能删了重建。"""
    async with SessionLocal() as session:
        agent = await session.scalar(select(Agent).where(Agent.id == AG_CUSTOM))
        if agent is None:
            agent = Agent(id=AG_CUSTOM, name="Custom 测试", created_at=1)
            session.add(agent)
        agent.name = "Custom 测试"
        agent.avatar = "🧪"
        agent.description = "custom adapter 测试"
        agent.capabilities = ["test"]
        agent.system_prompt = system_prompt
        agent.adapter_name = "custom"
        agent.model_provider = "deepseek"
        agent.model_id = "deepseek-chat"
        agent.api_key = api_key
        agent.api_base_url = None
        agent.tool_names = []
        agent.is_builtin = False
        agent.is_orchestrator = False
        agent.supports_vision = False
        await session.commit()


async def _create_conversation(client: AsyncClient) -> dict:
    res = await client.post(
        "/api/conversations", json={"mode": "single", "agentIds": [AG_CUSTOM]}
    )
    assert res.status_code == 201, res.text
    return res.json()["conversation"]


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch):
    for key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ARK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


async def _reset_settings(client: AsyncClient):
    await client.patch(
        "/api/settings",
        json={
            "deepseekApiKey": None,
            "openaiApiKey": None,
            "arkApiKey": None,
            "anthropicApiKey": None,
        },
    )


async def test_custom_agent_end_to_end(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    await _reset_settings(client)
    await _ensure_custom_agent(api_key="sk-agent-level")
    capture = _Capture(replies=["第一轮回答"])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    conv = await _create_conversation(client)
    res = await client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "你好"}
    )
    assert res.status_code == 202, res.text
    await wait_for_runs(res.json()["runIds"])

    # per-agent key 优先，不需要任何全局配置
    assert capture.client_args == [("deepseek", "sk-agent-level", None)]

    # system prompt：workspace 信息块在前，agent systemPrompt 在后
    system_content = capture.request_messages[0][0]["content"]
    assert system_content.startswith("<workspace_info>")
    assert "你是测试 agent" in system_content

    # 回复落库 + usage 落库
    agent_messages = await messages_of(conv["id"], role="agent")
    assert len(agent_messages) == 1
    assert agent_messages[0].parts[0]["type"] == "text"
    assert agent_messages[0].parts[0]["content"] == "第一轮回答"
    assert agent_messages[0].usage["inputTokens"] == 11
    assert agent_messages[0].usage["outputTokens"] == 7

    run = await run_row(res.json()["runIds"][0])
    assert run.status == "complete"
    assert run.usage["inputTokens"] == 11
    assert run.usage["model"] == "deepseek-chat"


async def test_global_settings_key_used_when_agent_key_missing(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    await _reset_settings(client)
    await _ensure_custom_agent(api_key=None)
    await client.patch("/api/settings", json={"deepseekApiKey": "sk-global"})

    capture = _Capture(replies=["好"])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    conv = await _create_conversation(client)
    res = await client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "在吗"}
    )
    await wait_for_runs(res.json()["runIds"])

    assert capture.client_args == [("deepseek", "sk-global", None)]


async def test_env_key_used_when_nothing_else_set(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    await _reset_settings(client)
    await _ensure_custom_agent(api_key=None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")

    capture = _Capture(replies=["好"])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    conv = await _create_conversation(client)
    res = await client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "在吗"}
    )
    await wait_for_runs(res.json()["runIds"])

    assert capture.client_args == [("deepseek", "sk-env", None)]


async def test_history_injected_on_second_turn(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    await _reset_settings(client)
    await _ensure_custom_agent(api_key="sk-agent-level", system_prompt="你是测试 agent")
    capture = _Capture(replies=["苹果", "你刚才说的是苹果"])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    conv = await _create_conversation(client)

    res1 = await client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "记住：我喜欢苹果"}
    )
    await wait_for_runs(res1.json()["runIds"])
    res2 = await client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "我喜欢什么？"}
    )
    await wait_for_runs(res2.json()["runIds"])

    # 第二轮请求的 messages：system → 历史（user + assistant）→ 当前 user
    # （run 结束后末尾会追加本轮 assistant，只看初始形状的前 4 条）
    second = capture.request_messages[1]
    assert [m["role"] for m in second[:4]] == ["system", "user", "assistant", "user"]
    assert second[1]["content"] == "记住：我喜欢苹果"
    assert second[2]["content"] == "苹果"
    assert second[3]["content"] == "我喜欢什么？"


async def test_group_chat_system_note_and_history(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
):
    from tests.test_agent_runner import _ensure_second_mock_agent

    await _reset_settings(client)
    await _ensure_custom_agent(api_key="sk-agent-level")
    capture = _Capture(replies=["群聊回复", "第二轮"])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    # 群聊：custom agent + mock agent（不发言，只凑 agentIds > 1）
    second_mock = await _ensure_second_mock_agent()
    res = await client.post(
        "/api/conversations",
        json={"mode": "group", "agentIds": [AG_CUSTOM, second_mock]},
    )
    conv = res.json()["conversation"]

    # 先让 custom agent 说一句，再问第二轮，看历史注入
    res1 = await client.post(
        f"/api/conversations/{conv['id']}/messages",
        json={"content": "自我介绍", "mentionedAgentIds": [AG_CUSTOM]},
    )
    await wait_for_runs(res1.json()["runIds"])
    res2 = await client.post(
        f"/api/conversations/{conv['id']}/messages",
        json={"content": "继续", "mentionedAgentIds": [AG_CUSTOM]},
    )
    await wait_for_runs(res2.json()["runIds"])

    # 群聊（agentIds > 1）→ system prompt 末尾带群聊上下文说明
    system_content = capture.request_messages[-1][0]["content"]
    assert "## 群聊上下文" in system_content

    # 历史：自己的上一条以 assistant 角色注入
    last_messages = capture.request_messages[-1]
    roles_contents = [(m["role"], m["content"]) for m in last_messages]
    assert ("assistant", "群聊回复") in roles_contents
