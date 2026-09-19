"""fs_write 审批流全链路（fake LLM → 真工具 → 挂起 → HTTP 审批 → run 收敛）。

后续 bash / ask_user 的审批用例也放这个文件（同一套夹具）。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.adapters import custom as custom_module
from app.db.models import Agent, Workspace
from app.db.session import SessionLocal
from app.services.pending_questions import pending_questions
from app.services.pending_writes import pending_writes
from app.tools.ask_user import ASK_USER_TOOL
from app.tools.fs_write import FS_WRITE_TOOL
from app.tools.registry import tool_registry
from tests.helpers import EventCollector, wait_for_runs

AG_TOOLS = "ag_tools_test"


# ─── 假 OpenAI 客户端（支持工具调用轮次） ──────────────────


class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = iter(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _chunk(content: str | None = None, tool_calls: list | None = None, finish_reason: str | None = None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None)


def _tool_delta(index: int, call_id: str, name: str, arguments: str):
    return SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _usage_chunk(p: int, c: int):
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c))


class _ToolCapture:
    """第一轮让模型调工具，之后的轮次按给定回复串行返回。记录每轮请求的 messages。"""

    def __init__(self, tool_name: str, tool_args: dict, final_replies: list[str]) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.final_replies = list(final_replies)
        self.request_messages: list[list[dict]] = []

    def build(self, provider, override_key, api_base_url):
        capture = self

        class _Completions:
            async def create(self, **kwargs):
                capture.request_messages.append(kwargs["messages"])
                if kwargs["messages"][-1].get("role") != "tool":
                    reply = None
                    return _FakeStream(
                        [
                            _chunk(
                                tool_calls=[
                                    _tool_delta(
                                        0, "call_1", capture.tool_name, json.dumps(capture.tool_args)
                                    )
                                ],
                                finish_reason="tool_calls",
                            ),
                            _usage_chunk(10, 5),
                        ]
                    )
                reply = capture.final_replies.pop(0)
                return _FakeStream([_chunk(content=reply, finish_reason="stop"), _usage_chunk(10, 5)])

        return SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))


# ─── 夹具 ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _register_fs_write():
    tool_registry.register(FS_WRITE_TOOL)
    tool_registry.register(ASK_USER_TOOL)
    yield
    tool_registry._tools.pop(FS_WRITE_TOOL.name, None)
    tool_registry._tools.pop(ASK_USER_TOOL.name, None)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch):
    for key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ARK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


async def _ensure_tools_agent(tool_names: list[str]) -> None:
    async with SessionLocal() as session:
        agent = await session.scalar(select(Agent).where(Agent.id == AG_TOOLS))
        if agent is None:
            agent = Agent(id=AG_TOOLS, name="工具测试", created_at=1)
            session.add(agent)
        agent.name = "工具测试"
        agent.avatar = "🛠"
        agent.description = "审批流测试"
        agent.capabilities = []
        agent.system_prompt = "你是测试 agent"
        agent.adapter_name = "custom"
        agent.model_provider = "deepseek"
        agent.model_id = "deepseek-chat"
        agent.api_key = "sk-agent-level"
        agent.api_base_url = None
        agent.tool_names = tool_names
        agent.is_builtin = False
        agent.is_orchestrator = False
        agent.supports_vision = False
        await session.commit()


async def _conversation_with_agent(client: AsyncClient, tool_names: list[str] | None = None) -> dict:
    await _ensure_tools_agent(tool_names or ["fs_write"])
    res = await client.post("/api/conversations", json={"mode": "single", "agentIds": [AG_TOOLS]})
    assert res.status_code == 201, res.text
    return res.json()["conversation"]


async def _workspace_root(conversation_id: str) -> str:
    async with SessionLocal() as session:
        ws = await session.scalar(select(Workspace).where(Workspace.conversation_id == conversation_id))
    assert ws is not None
    return ws.root_path


async def _send_and_wait_pending(client: AsyncClient, conv_id: str) -> dict:
    """发消息 → 等 fs_write.pending 出现，返回 pending payload 与 runIds。"""
    res = await client.post(f"/api/conversations/{conv_id}/messages", json={"content": "写个文件"})
    assert res.status_code == 202, res.text
    run_ids = res.json()["runIds"]

    deadline = asyncio.get_event_loop().time() + 10
    while asyncio.get_event_loop().time() < deadline:
        items = pending_writes.list_by_conversation(conv_id)
        if items:
            return {"pending": items[0], "runIds": run_ids}
        await asyncio.sleep(0.02)
    raise AssertionError("pending write never appeared")


# ─── 用例 ─────────────────────────────────────────────────


async def test_review_mode_approve_writes_file(client: AsyncClient, monkeypatch):
    conv = await _conversation_with_agent(client)
    capture = _ToolCapture("fs_write", {"path": "notes.txt", "content": "审批后的内容"}, ["写好了"])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    async with EventCollector() as collector:
        started = await _send_and_wait_pending(client, conv["id"])
        pending = started["pending"]
        assert pending["path"] == "notes.txt"
        assert pending["oldContent"] is None
        assert pending["newContent"] == "审批后的内容"

        res = await client.post(
            f"/api/conversations/{conv['id']}/pending-writes/{pending['id']}",
            json={"action": "approve"},
        )
        assert res.status_code == 200, res.text

        await wait_for_runs(started["runIds"])

    root = await _workspace_root(conv["id"])
    with open(f"{root}/notes.txt") as fh:
        assert fh.read() == "审批后的内容"

    resolved = collector.of_type("fs_write.resolved")
    assert resolved[-1].pendingId == pending["id"]
    assert resolved[-1].applied is True

    # 工具结果回灌给 LLM（第二轮请求的 tool 消息）
    tool_msgs = [m for m in capture.request_messages[-1] if m.get("role") == "tool"]
    assert json.loads(tool_msgs[0]["content"])["applied"] == "review"

    # 审批后 store 清空
    assert pending_writes.list_by_conversation(conv["id"]) == []


async def test_review_mode_reject_keeps_file_unchanged(client: AsyncClient, monkeypatch):
    conv = await _conversation_with_agent(client)
    root = await _workspace_root(conv["id"])
    with open(f"{root}/existing.txt", "w") as fh:
        fh.write("旧内容")

    capture = _ToolCapture("fs_write", {"path": "existing.txt", "content": "新内容"}, ["好的，不改了"])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    async with EventCollector() as collector:
        started = await _send_and_wait_pending(client, conv["id"])
        pending = started["pending"]
        assert pending["oldContent"] == "旧内容"

        res = await client.post(
            f"/api/conversations/{conv['id']}/pending-writes/{pending['id']}",
            json={"action": "reject"},
        )
        assert res.status_code == 200, res.text
        await wait_for_runs(started["runIds"])

    with open(f"{root}/existing.txt") as fh:
        assert fh.read() == "旧内容"

    resolved = collector.of_type("fs_write.resolved")
    assert resolved[-1].applied is False

    # 拒绝以错误形式回灌给 LLM
    tool_msgs = [m for m in capture.request_messages[-1] if m.get("role") == "tool"]
    payload = json.loads(tool_msgs[0]["content"])
    assert payload == {"error": "User rejected the file change"}


async def test_auto_mode_writes_immediately(client: AsyncClient, monkeypatch):
    conv = await _conversation_with_agent(client)
    res = await client.patch(
        f"/api/conversations/{conv['id']}", json={"fsWriteApprovalMode": "auto"}
    )
    assert res.status_code == 200, res.text

    capture = _ToolCapture("fs_write", {"path": "auto.txt", "content": "直写"}, ["done"])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    res = await client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "写"})
    assert res.status_code == 202, res.text
    await wait_for_runs(res.json()["runIds"])

    root = await _workspace_root(conv["id"])
    with open(f"{root}/auto.txt") as fh:
        assert fh.read() == "直写"
    assert pending_writes.list_by_conversation(conv["id"]) == []


async def test_abort_cancels_pending_write(client: AsyncClient, monkeypatch):
    conv = await _conversation_with_agent(client)
    capture = _ToolCapture("fs_write", {"path": "never.txt", "content": "不该落盘"}, [""])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    started = await _send_and_wait_pending(client, conv["id"])

    res = await client.post(f"/api/runs/{started['runIds'][0]}/abort")
    assert res.status_code == 200, res.text
    await wait_for_runs(started["runIds"])

    root = await _workspace_root(conv["id"])
    import os

    assert not os.path.exists(f"{root}/never.txt")
    assert pending_writes.list_by_conversation(conv["id"]) == []

    run = started["runIds"][0]
    from tests.helpers import run_row

    row = await run_row(run)
    assert row.status == "aborted"


# ─── ask_user ─────────────────────────────────────────────


async def test_ask_user_answered_and_fed_back(client: AsyncClient, monkeypatch):
    conv = await _conversation_with_agent(client, tool_names=["ask_user"])
    questions = [
        {
            "question": "用哪个方案？",
            "header": "方案",
            "options": [{"label": "A"}, {"label": "B"}],
        },
        {
            "question": "还有什么补充？",
            "header": "备注",
            "options": [{"label": "无"}, {"label": "有"}],
        },
    ]
    capture = _ToolCapture("ask_user", {"questions": questions}, ["收到，按 A 方案来"])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    async with EventCollector() as collector:
        res = await client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "问我"})
        assert res.status_code == 202, res.text
        run_ids = res.json()["runIds"]

        deadline = asyncio.get_event_loop().time() + 10
        pending = None
        while asyncio.get_event_loop().time() < deadline:
            items = pending_questions.list_by_conversation(conv["id"])
            if items:
                pending = items[0]
                break
            await asyncio.sleep(0.02)
        assert pending is not None
        assert pending["questions"][0]["question"] == "用哪个方案？"

        res = await client.post(
            f"/api/conversations/{conv['id']}/pending-questions/{pending['id']}",
            json={
                "answers": {
                    "用哪个方案？": {"selectedLabels": ["A"]},
                    "还有什么补充？": {"selectedLabels": ["无"], "freeformNote": "尽快开始"},
                }
            },
        )
        assert res.status_code == 200, res.text
        await wait_for_runs(run_ids)

    resolved = collector.of_type("ask_user.resolved")
    assert resolved[-1].answered is True
    assert pending_questions.list_by_conversation(conv["id"]) == []

    # 答案格式化后回灌给 LLM
    tool_msgs = [m for m in capture.request_messages[-1] if m.get("role") == "tool"]
    answers = json.loads(tool_msgs[0]["content"])["answers"]
    assert answers["用哪个方案？"] == "A"
    assert answers["还有什么补充？"] == "无 ; note: 尽快开始"


async def test_ask_user_aborted_returns_error(client: AsyncClient, monkeypatch):
    conv = await _conversation_with_agent(client, tool_names=["ask_user"])
    questions = [{"question": "继续吗？", "header": "确认", "options": [{"label": "是"}, {"label": "否"}]}]
    capture = _ToolCapture("ask_user", {"questions": questions}, ["好"])
    monkeypatch.setattr(custom_module, "build_client", capture.build)

    res = await client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "问"})
    run_ids = res.json()["runIds"]

    deadline = asyncio.get_event_loop().time() + 10
    while asyncio.get_event_loop().time() < deadline:
        if pending_questions.list_by_conversation(conv["id"]):
            break
        await asyncio.sleep(0.02)

    abort_res = await client.post(f"/api/runs/{run_ids[0]}/abort")
    assert abort_res.status_code == 200, abort_res.text
    await wait_for_runs(run_ids)

    assert pending_questions.list_by_conversation(conv["id"]) == []
    # abort 路径工具以错误收场，模型不会再有第二轮回复需求（capture 的最终回复未消费也没关系）
