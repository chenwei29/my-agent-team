"""产物 / 部署的事件接线：adapter 发 artifact.create 与 deploy.status，runner 注入对应 part。

分两段验证：
1. adapter 层 —— 工具成功后是否发出内容完整的事件（事件里带整条产物记录）
2. runner 层 —— 收到事件后是否把 artifact_ref / deploy_status 追加到流式消息并补发 part.start
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import AsyncClient

from app.adapters import custom as custom_module
from app.adapters.types import AdapterInput
from app.db.models import Artifact, Message
from app.db.session import SessionLocal
from app.schemas.events import (
    ArtifactCreateEvent,
    ArtifactRecord,
    DeployStatusEvent,
    MessageEndEvent,
    MessageStartEvent,
    StreamEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.services import agent_runner as runner_module
from app.tools.registry import tool_registry
from app.tools.types import ToolDef, ToolResult
from app.utils.abort import AbortSignal
from app.utils.ids import new_artifact_id
from app.utils.time import now_ms
from tests.conftest import MOCK_AGENT_ID
from tests.helpers import EventCollector, wait_for_conversation_runs

DEPLOYMENT = {
    "id": "dep_TestDeploy01",
    "artifactId": "art_x",
    "title": "页面",
    "version": 1,
    "previewPath": "/api/deployments/dep_TestDeploy01",
    "status": "ready",
    "sourceType": "artifact",
    "createdAt": 1,
}


def _adapter_input(names: list[str]) -> AdapterInput:
    return AdapterInput(
        agentId="ag_test",
        conversationId="conv_test",
        runId="run_test",
        prompt="做事",
        workspacePath="/tmp",
        systemPrompt="你是测试 agent",
        apiKey="k",
        apiBaseUrl="https://example.com/v1",
        modelId="m",
        toolNames=names,
        customConfig={"modelProvider": "openai-compatible", "supportsVision": False},
    )


async def _run_tool(monkeypatch: pytest.MonkeyPatch, name: str, tool: ToolDef):
    """把 stub 工具注册到 name 上跑一遍 adapter 的工具循环（不联网：直接喂两轮 chunk）。"""
    from tests.test_custom_adapter import _chunk, _install, _tool_delta

    original = tool_registry._tools.get(name)
    tool_registry.register(tool)
    try:
        _install(
            monkeypatch,
            [
                [
                    _chunk(
                        tool_calls=[_tool_delta(0, "call_1", name, "{}")],
                        finish_reason="tool_calls",
                    )
                ],
                [_chunk(content="好了", finish_reason="stop")],
            ],
        )
        adapter = custom_module.CustomAgentAdapter()
        return [
            event
            async for event in adapter.stream(_adapter_input([name]), AbortSignal())
        ]
    finally:
        if original is not None:
            tool_registry.register(original)


async def test_write_artifact_tool_emits_full_artifact_event(
    client: AsyncClient, conversation: dict, monkeypatch: pytest.MonkeyPatch
):
    conversation_id = conversation["id"]
    artifact_id = new_artifact_id()

    async def handler(_args: dict, _ctx: Any) -> ToolResult:
        async with SessionLocal() as session:
            session.add(
                Artifact(
                    id=artifact_id,
                    conversation_id=conversation_id,
                    type="document",
                    title="事件产物",
                    content={"type": "document", "format": "markdown", "content": "x"},
                    version=2,
                    parent_artifact_id="art_parent",
                    created_by_agent_id=MOCK_AGENT_ID,
                    created_at=now_ms(),
                )
            )
            await session.commit()
        return ToolResult(ok=True, value={"artifactId": artifact_id})

    events = await _run_tool(
        monkeypatch,
        "write_artifact",
        ToolDef(name="write_artifact", description="", parameters={}, handler=handler),
    )

    created = [e for e in events if e.type == "artifact.create"]
    assert len(created) == 1
    record = created[0].artifact
    assert record.id == artifact_id
    assert record.conversationId == conversation_id
    assert record.title == "事件产物"
    assert record.version == 2
    assert record.parentArtifactId == "art_parent"
    assert record.content["type"] == "document"
    # 事件排在 tool.result 之后、message.end 之前
    types = [e.type for e in events]
    assert types.index("tool.result") < types.index("artifact.create") < types.index("message.end")


async def test_write_artifact_failure_emits_nothing(monkeypatch: pytest.MonkeyPatch):
    async def handler(_args: dict, _ctx: Any) -> ToolResult:
        return ToolResult(ok=False, error="Invalid content")

    events = await _run_tool(
        monkeypatch,
        "write_artifact",
        ToolDef(name="write_artifact", description="", parameters={}, handler=handler),
    )
    assert "artifact.create" not in [e.type for e in events]


async def test_deploy_tool_emits_deploy_status(monkeypatch: pytest.MonkeyPatch):
    async def handler(_args: dict, _ctx: Any) -> ToolResult:
        return ToolResult(ok=True, value=dict(DEPLOYMENT))

    events = await _run_tool(
        monkeypatch,
        "deploy_artifact",
        ToolDef(name="deploy_artifact", description="", parameters={}, handler=handler),
    )

    deploy_events = [e for e in events if e.type == "deploy.status"]
    assert len(deploy_events) == 1
    assert deploy_events[0].deployment == DEPLOYMENT
    assert deploy_events[0].messageId  # 挂到产出它的那条 agent 消息上


# ─── runner 侧注入 ─────────────────────────────────────────


class _FakeAdapter:
    """手搓一段工具循环事件，专门用来验证 runner 的 part 注入。"""

    name = "fake"

    async def stream(self, input: AdapterInput, signal: AbortSignal) -> AsyncIterator[StreamEvent]:
        message_id = "msg_fake_events"
        stamp = now_ms()
        yield MessageStartEvent(
            conversationId=input.conversationId,
            timestamp=stamp,
            messageId=message_id,
            agentId=input.agentId,
            runId=input.runId,
        )
        yield ToolCallEvent(
            conversationId=input.conversationId,
            timestamp=stamp,
            messageId=message_id,
            callId="call_1",
            toolName="write_artifact",
            args={},
        )
        yield ToolResultEvent(
            conversationId=input.conversationId,
            timestamp=stamp,
            messageId=message_id,
            callId="call_1",
            result={"artifactId": "art_1"},
            isError=False,
        )
        yield ArtifactCreateEvent(
            conversationId=input.conversationId,
            timestamp=stamp,
            artifact=ArtifactRecord(
                id="art_1",
                conversationId=input.conversationId,
                type="document",
                title="注入产物",
                content={"type": "document", "format": "markdown", "content": "x"},
                version=1,
                createdByAgentId=input.agentId,
                createdAt=stamp,
            ),
        )
        yield DeployStatusEvent(
            conversationId=input.conversationId,
            timestamp=stamp,
            messageId=message_id,
            deployment=dict(DEPLOYMENT),
        )
        yield MessageEndEvent(
            conversationId=input.conversationId, timestamp=stamp, messageId=message_id
        )


async def test_runner_injects_artifact_ref_and_deploy_status_parts(
    client: AsyncClient, conversation: dict, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(runner_module, "_MOCK_ADAPTER", _FakeAdapter())

    async with EventCollector() as collector:
        res = await client.post(
            f"/api/conversations/{conversation['id']}/messages", json={"content": "产个东西再部署"}
        )
        assert res.status_code == 202, res.text
        await wait_for_conversation_runs(conversation["id"])
        await asyncio.sleep(0.05)  # 等事件被消费

    async with SessionLocal() as session:
        row = await session.get(Message, "msg_fake_events")
    assert [part["type"] for part in row.parts] == [
        "tool_use",
        "tool_result",
        "artifact_ref",
        "deploy_status",
    ]
    assert row.parts[2] == {"type": "artifact_ref", "artifactId": "art_1"}
    assert row.parts[3] == {"type": "deploy_status", "deployment": DEPLOYMENT}

    # 落库之后补发 part.start，两个 part 的下标与数组位置一致
    starts = [
        e
        for e in collector.events
        if e.type == "part.start" and e.messageId == "msg_fake_events"
    ]
    assert [(e.partIndex, e.part.type) for e in starts] == [
        (2, "artifact_ref"),
        (3, "deploy_status"),
    ]
    # 原始事件照常广播
    assert len(collector.of_type("artifact.create")) == 1
    assert len(collector.of_type("deploy.status")) == 1
