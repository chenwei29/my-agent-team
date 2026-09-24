"""Agent 草稿生成（build_heuristic_agent_config_draft + 请求校验）。

关键词 → 工具预设的映射用例锁死四种典型描述；生成的草稿一律是
non-orchestrator 的 custom agent（deepseek 默认模型、不含 Orchestrator 专用工具）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.entities import AgentDraftBody
from app.services.agent_draft import create_agent_config_draft


def test_maps_local_code_intent_to_workspace_tools():
    draft = create_agent_config_draft("我想创建一个能修改本地仓库代码、运行测试并修复 bug 的 Agent")

    assert draft["name"] == "代码工程师"
    assert "fs_write" in draft["toolNames"]
    assert "bash" in draft["toolNames"]
    assert "deploy_workspace" in draft["toolNames"]
    assert "write_artifact" not in draft["toolNames"]
    assert [s["toolName"] for s in draft["toolPermissionSummaries"]] == draft["toolNames"]


def test_maps_review_intent_to_read_tools_without_write_permission():
    draft = create_agent_config_draft("帮我审查验证已有代码和产物，发现风险并给出修改建议")

    assert draft["name"] == "审查验证助手"
    assert "fs_read" in draft["toolNames"]
    assert "bash" in draft["toolNames"]
    assert "fs_write" not in draft["toolNames"]
    assert "write_artifact" not in draft["toolNames"]


def test_keeps_generated_drafts_as_non_orchestrator_custom_agents():
    draft = create_agent_config_draft("创建一个负责拆任务调度其他 agent 的 plan_tasks orchestrator")

    assert draft["adapterName"] == "custom"
    assert draft["modelProvider"] == "deepseek"
    assert draft["modelId"] == "deepseek-v4-flash"
    assert "plan_tasks" not in draft["toolNames"]


def test_rejects_invalid_draft_requests_at_the_schema_boundary():
    with pytest.raises(ValidationError):
        AgentDraftBody.model_validate({"intent": ""})
    body = AgentDraftBody.model_validate({"intent": "做一个代码助手"})
    assert body.intent == "做一个代码助手"


# ─── POST /api/agents/draft 契约 ───────────────────────────


async def test_draft_route_returns_draft_envelope(client):
    res = await client.post(
        "/api/agents/draft", json={"intent": "帮我搭建一个生成网页原型的工具", "followUp": "偏现代风格"}
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert set(body) == {"draft"}
    draft = body["draft"]
    assert draft["adapterName"] == "custom"
    assert draft["modelProvider"] == "deepseek"
    assert draft["supportsVision"] is True
    assert draft["systemPrompt"].startswith("你是 ")
    assert draft["rationale"]
    assert draft["assumptions"]
    assert draft["toolPermissionSummaries"]


async def test_draft_route_rejects_short_intent(client):
    res = await client.post("/api/agents/draft", json={"intent": "短"})
    assert res.status_code == 400
    body = res.json()
    assert body["error"] == "Invalid body"
    assert body["issues"]
