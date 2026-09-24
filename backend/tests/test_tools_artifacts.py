"""write_artifact / read_artifact / plan_tasks 工具的最小语义。

走真实 DB（FK 开启），所以用 API 建好会话与 mock agent 再操作。
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.db.models import Artifact
from app.db.session import SessionLocal
from app.tools.artifacts import READ_ARTIFACT_TOOL, WRITE_ARTIFACT_TOOL
from app.tools.plan_tasks import PLAN_TASKS_TOOL
from app.tools.types import ToolContext

MOCK_ID = "ag_e2e_mock"


def _ctx(conversation_id: str, agent_id: str) -> ToolContext:
    return ToolContext(
        conversation_id=conversation_id,
        agent_id=agent_id,
        run_id="run_test",
        workspace_path=".",
        abort_signal=None,
    )


@pytest.fixture(autouse=True)
async def conversation_with_agent(client: AsyncClient) -> dict:
    res = await client.post("/api/conversations", json={"mode": "single", "agentIds": [MOCK_ID]})
    assert res.status_code == 201, res.text
    return res.json()["conversation"]


async def _artifacts_of(conversation_id: str) -> list[Artifact]:
    async with SessionLocal() as session:
        return list(
            (
                await session.execute(
                    select(Artifact).where(Artifact.conversation_id == conversation_id)
                )
            )
            .scalars()
            .all()
        )


async def test_write_artifact_creates_row(conversation_with_agent: dict):
    conv = conversation_with_agent
    result = await WRITE_ARTIFACT_TOOL.handler(
        {
            "type": "document",
            "title": "说明文档",
            "content": {"format": "markdown", "content": "# 标题"},
        },
        _ctx(conv["id"], MOCK_ID),
    )
    assert result.ok
    assert result.value["artifactId"].startswith("art_")
    assert result.value["version"] == 1
    assert result.value["parentArtifactId"] is None

    rows = await _artifacts_of(conv["id"])
    assert len(rows) == 1
    assert rows[0].created_by_agent_id == MOCK_ID
    # 内容经规整后带上 type 判别键
    assert rows[0].content == {"type": "document", "format": "markdown", "content": "# 标题"}


async def test_write_artifact_version_chain(conversation_with_agent: dict):
    conv = conversation_with_agent
    first = await WRITE_ARTIFACT_TOOL.handler(
        {"type": "document", "title": "v1", "content": {"format": "markdown", "content": "a"}},
        _ctx(conv["id"], MOCK_ID),
    )
    second = await WRITE_ARTIFACT_TOOL.handler(
        {
            "type": "document",
            "title": "v2",
            "content": {"format": "markdown", "content": "b"},
            "parentArtifactId": first.value["artifactId"],
        },
        _ctx(conv["id"], MOCK_ID),
    )
    assert second.ok
    assert second.value["version"] == 2
    assert second.value["parentArtifactId"] == first.value["artifactId"]


async def test_write_artifact_parent_scoped_to_conversation(conversation_with_agent: dict):
    conv = conversation_with_agent
    first = await WRITE_ARTIFACT_TOOL.handler(
        {"type": "document", "title": "v1", "content": {"format": "markdown", "content": "a"}},
        _ctx(conv["id"], MOCK_ID),
    )
    # 换个会话引用同一个 parent → 拒绝
    result = await WRITE_ARTIFACT_TOOL.handler(
        {
            "type": "document",
            "title": "盗版",
            "content": {"format": "markdown", "content": "x"},
            "parentArtifactId": first.value["artifactId"],
        },
        _ctx("conv_does_not_exist", MOCK_ID),
    )
    assert not result.ok
    assert "different conversation" in result.error


async def test_write_artifact_parent_not_found(conversation_with_agent: dict):
    result = await WRITE_ARTIFACT_TOOL.handler(
        {
            "type": "document",
            "title": "孤儿",
            "content": {"format": "markdown", "content": "x"},
            "parentArtifactId": "art_nope",
        },
        _ctx(conversation_with_agent["id"], MOCK_ID),
    )
    assert not result.ok
    assert "not found: art_nope" in result.error


async def test_write_artifact_invalid_content(conversation_with_agent: dict):
    result = await WRITE_ARTIFACT_TOOL.handler(
        {"type": "document", "title": "t", "content": 123},
        _ctx(conversation_with_agent["id"], MOCK_ID),
    )
    assert not result.ok
    assert result.error == "Invalid content for type document"

    # 字符串化的 content 包装会被规整层解包救回
    import json as _json

    result = await WRITE_ARTIFACT_TOOL.handler(
        {"type": "document", "title": "t", "content": _json.dumps({"content": "# hi"})},
        _ctx(conversation_with_agent["id"], MOCK_ID),
    )
    assert result.ok


async def test_write_artifact_invalid_type(conversation_with_agent: dict):
    result = await WRITE_ARTIFACT_TOOL.handler(
        {"type": "video", "title": "t", "content": {}},
        _ctx(conversation_with_agent["id"], MOCK_ID),
    )
    assert not result.ok
    assert result.error.startswith("Invalid args:")


async def test_read_artifact_scoped_to_conversation(conversation_with_agent: dict):
    conv = conversation_with_agent
    created = await WRITE_ARTIFACT_TOOL.handler(
        {"type": "document", "title": "doc", "content": {"format": "markdown", "content": "hi"}},
        _ctx(conv["id"], MOCK_ID),
    )
    artifact_id = created.value["artifactId"]

    result = await READ_ARTIFACT_TOOL.handler(
        {"artifactId": artifact_id}, _ctx(conv["id"], MOCK_ID)
    )
    assert result.ok
    assert result.value["id"] == artifact_id
    assert result.value["content"] == {"type": "document", "format": "markdown", "content": "hi"}

    # 其他会话读不到
    result = await READ_ARTIFACT_TOOL.handler(
        {"artifactId": artifact_id}, _ctx("conv_other", MOCK_ID)
    )
    assert not result.ok
    assert result.error == f"Artifact not found: {artifact_id}"


async def test_plan_tasks_ack(conversation_with_agent: dict):
    result = await PLAN_TASKS_TOOL.handler(
        {
            "reasoning": "两步走",
            "tasks": [
                {
                    "id": "t1",
                    "agentId": MOCK_ID,
                    "task": "写代码",
                    "taskKind": "code",
                    "expectedOutputs": [{"id": "o1", "type": "project", "required": True}],
                },
                {
                    "id": "t2",
                    "agentId": MOCK_ID,
                    "task": "审查",
                    "dependsOn": ["t1"],
                    "acceptanceCriteria": ["测试全绿"],
                },
            ],
        },
        _ctx(conversation_with_agent["id"], MOCK_ID),
    )
    assert result.ok
    assert result.value == {"acknowledged": True, "taskCount": 2}


async def test_plan_tasks_invalid_plan(conversation_with_agent: dict):
    result = await PLAN_TASKS_TOOL.handler(
        {"reasoning": "没任务"}, _ctx(conversation_with_agent["id"], MOCK_ID)
    )
    assert not result.ok
    assert result.error.startswith("Invalid plan:")
