"""用户自建 Agent 的服务：创建、删除、改配置，以及各 adapter 的准入校验。

自建 Agent 默认 adapterName='custom'，也可选 Claude Code / Codex SDK adapter。
内置 Agent（is_builtin=True）不可被删除，但**可以**改配置。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Agent
from app.errors import NotFoundError, ServiceError
from app.services.openai_compatible import (
    validate_openai_compatible_api_key,
    validate_openai_compatible_base_url,
)
from app.utils.ids import new_agent_id
from app.utils.time import now_ms


async def list_agents_ordered(session: AsyncSession) -> list[Agent]:
    """内置在前，按 createdAt desc。

    只按 (isBuiltin desc, createdAt desc) 排时，5 个内置 agent 的 createdAt 是在同一毫秒
    求值的，组内顺序实际不确定。这里补一个 id asc 作稳定 tiebreak，避免侧栏每次刷新换序。
    """
    result = await session.scalars(
        select(Agent).order_by(Agent.is_builtin.desc(), Agent.created_at.desc(), Agent.id.asc())
    )
    return list(result)


async def create_custom_agent(session: AsyncSession, args: dict[str, Any]) -> Agent:
    agent_id = new_agent_id()
    created_at = now_ms()
    adapter_name: str = args.get("adapter_name") or "custom"

    if adapter_name == "custom":
        if not args.get("model_provider") or not args.get("model_id"):
            raise ServiceError("Custom adapter requires modelProvider and modelId")
        base_url_error = validate_openai_compatible_base_url(
            args.get("model_provider"), args.get("api_base_url")
        )
        if base_url_error:
            raise ServiceError(base_url_error)
        api_key_error = validate_openai_compatible_api_key(
            args.get("model_provider"), args.get("api_key")
        )
        if api_key_error:
            raise ServiceError(api_key_error)

    agent = Agent(
        id=agent_id,
        name=args["name"].strip(),
        avatar=(args.get("avatar") or "").strip() or "🤖",
        description=args["description"].strip(),
        capabilities=args.get("capabilities") or [],
        system_prompt=args["system_prompt"],
        adapter_name=adapter_name,
        model_provider=(args.get("model_provider") if adapter_name == "custom" else None),
        model_id=args.get("model_id"),
        api_key=(args.get("api_key") or "").strip() or None,
        api_base_url=(args.get("api_base_url") or "").strip() or None,
        # SDK adapter 走各自内置工具集，不消费 toolNames；强制空数组避免 UI 残留
        tool_names=(args.get("tool_names") or []) if adapter_name == "custom" else [],
        is_builtin=False,
        is_orchestrator=False,
        supports_vision=bool(args.get("supports_vision") or False),
        created_at=created_at,
    )
    session.add(agent)
    await session.commit()
    return agent


async def delete_custom_agent(session: AsyncSession, agent_id: str) -> None:
    agent = await session.scalar(select(Agent).where(Agent.id == agent_id))
    if agent is None:
        raise NotFoundError(f"Agent not found: {agent_id}")
    if agent.is_builtin:
        raise ServiceError("Built-in agents cannot be deleted")

    result = await session.execute(
        sa_delete(Agent).where(Agent.id == agent_id, Agent.is_builtin.is_(False))
    )
    await session.commit()
    if result.rowcount == 0:
        raise ServiceError(f"Failed to delete agent: {agent_id}")


async def update_custom_agent(
    session: AsyncSession, agent_id: str, patch: dict[str, Any], provided: set[str]
) -> Agent:
    """patch 里「没出现的键」与「显式 null」语义不同，靠 provided（model_fields_set）区分。"""
    agent = await session.scalar(select(Agent).where(Agent.id == agent_id))
    if agent is None:
        raise NotFoundError(f"Agent not found: {agent_id}")
    # 内建 agent 允许修改配置（API key / system prompt / model 等），但删除仍受保护

    next_adapter_name = patch.get("adapter_name") or agent.adapter_name
    # modelProvider / modelId 走「空值回落」语义：显式 null 也回落到 agent 已有值（只参与校验，不改写）
    next_model_provider = (
        patch.get("model_provider") if "model_provider" in provided else None
    ) or agent.model_provider
    next_model_id = (patch.get("model_id") if "model_id" in provided else None) or agent.model_id
    next_api_base_url = (
        (patch.get("api_base_url") or "").strip() or None
        if "api_base_url" in provided
        else agent.api_base_url
    )
    next_api_key = (
        (patch.get("api_key") or "").strip() or None
        if "api_key" in provided
        else agent.api_key
    )

    if next_adapter_name == "custom" and (not next_model_provider or not next_model_id):
        raise ServiceError("Custom adapter requires modelProvider and modelId")
    if next_adapter_name == "custom":
        base_url_error = validate_openai_compatible_base_url(next_model_provider, next_api_base_url)
        if base_url_error:
            raise ServiceError(base_url_error)
        api_key_error = validate_openai_compatible_api_key(next_model_provider, next_api_key)
        if api_key_error:
            raise ServiceError(api_key_error)

    updates: dict[str, Any] = {}
    if "name" in provided:
        updates["name"] = (patch.get("name") or "").strip()
    if "description" in provided:
        updates["description"] = (patch.get("description") or "").strip()
    if "capabilities" in provided:
        updates["capabilities"] = patch.get("capabilities")
    if "system_prompt" in provided:
        updates["system_prompt"] = patch.get("system_prompt")
    if "adapter_name" in provided:
        updates["adapter_name"] = patch.get("adapter_name")
    if "model_id" in provided:
        updates["model_id"] = (patch.get("model_id") or "").strip() or None
    if "supports_vision" in provided:
        updates["supports_vision"] = patch.get("supports_vision")
    if "api_key" in provided:
        updates["api_key"] = (patch.get("api_key") or "").strip() or None
    if "api_base_url" in provided:
        updates["api_base_url"] = (patch.get("api_base_url") or "").strip() or None

    if next_adapter_name == "custom":
        if "model_provider" in provided:
            updates["model_provider"] = patch.get("model_provider")
        if "tool_names" in provided:
            updates["tool_names"] = patch.get("tool_names")
    else:
        # SDK adapter 走各自内置工具集，不消费 modelProvider / toolNames
        if "adapter_name" in provided and "model_id" not in provided:
            updates["model_id"] = None
        if provided & {"adapter_name", "model_provider", "tool_names"}:
            updates["model_provider"] = None
            updates["tool_names"] = []

    if not updates:
        return agent

    for key, value in updates.items():
        setattr(agent, key, value)
    await session.commit()
    return agent
