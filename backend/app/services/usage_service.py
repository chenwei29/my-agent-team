"""全局 token 用量聚合：今日 / 本周 / 全部 + 按会话 / 按 agent / 按模型分组。

口径（与会话内 UsageBadge 一致）：
- 每个 bucket 的 totalTokens = input + output + cacheRead + cacheCreation（含 cache 读写）；
- today / week 是**滚动窗口**（now-24h / now-7d），不是自然日历日；
- 分组依据：agent 按 run.agent_id，model 按 usage.model（缺 model 的 run 不进 byModel），
  会话按 run.conversation_id，top 会话按 totalTokens 取前 10。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Agent, AgentRun, Conversation
from app.utils.time import now_ms

_DAY_MS = 24 * 60 * 60 * 1000
_TOP_CONVERSATIONS = 10


def _empty_bucket() -> dict[str, int]:
    return {
        "inputTokens": 0,
        "outputTokens": 0,
        "cacheReadTokens": 0,
        "cacheCreationTokens": 0,
        "totalTokens": 0,
        "runs": 0,
    }


def _accumulate(bucket: dict[str, int], usage: dict[str, Any]) -> None:
    bucket["inputTokens"] += int(usage.get("inputTokens") or 0)
    bucket["outputTokens"] += int(usage.get("outputTokens") or 0)
    bucket["cacheReadTokens"] += int(usage.get("cacheReadTokens") or 0)
    bucket["cacheCreationTokens"] += int(usage.get("cacheCreationTokens") or 0)
    bucket["totalTokens"] += (
        int(usage.get("inputTokens") or 0)
        + int(usage.get("outputTokens") or 0)
        + int(usage.get("cacheReadTokens") or 0)
        + int(usage.get("cacheCreationTokens") or 0)
    )
    bucket["runs"] += 1


async def build_usage_summary(session: AsyncSession) -> dict[str, Any]:
    runs = (
        await session.scalars(select(AgentRun).where(AgentRun.usage.is_not(None)))
    ).all()

    now = now_ms()
    today_start = now - _DAY_MS
    week_start = now - 7 * _DAY_MS

    today = _empty_bucket()
    week = _empty_bucket()
    all_time = _empty_bucket()
    by_agent: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    by_conv: dict[str, dict[str, int]] = {}

    for run in runs:
        usage = run.usage
        if not usage:
            continue
        _accumulate(all_time, usage)
        if run.started_at >= week_start:
            _accumulate(week, usage)
        if run.started_at >= today_start:
            _accumulate(today, usage)

        _accumulate(by_agent.setdefault(run.agent_id, _empty_bucket()), usage)

        model = usage.get("model")
        if model:
            _accumulate(by_model.setdefault(model, _empty_bucket()), usage)

        _accumulate(by_conv.setdefault(run.conversation_id, _empty_bucket()), usage)

    agent_names: dict[str, str] = {}
    if by_agent:
        rows = await session.execute(
            select(Agent.id, Agent.name).where(Agent.id.in_(list(by_agent)))
        )
        agent_names = {r[0]: r[1] for r in rows.all()}

    top_conv_ids = [
        conv_id
        for conv_id, _ in sorted(
            by_conv.items(), key=lambda kv: kv[1]["totalTokens"], reverse=True
        )[:_TOP_CONVERSATIONS]
    ]
    conv_rows: dict[str, Conversation] = {}
    if top_conv_ids:
        conv_rows = {
            c.id: c
            for c in (
                await session.scalars(
                    select(Conversation).where(Conversation.id.in_(top_conv_ids))
                )
            ).all()
        }

    top_conversations = []
    for conv_id in top_conv_ids:
        conv = conv_rows.get(conv_id)
        bucket = by_conv.get(conv_id)
        if conv is None or bucket is None:
            continue
        top_conversations.append(
            {
                "id": conv_id,
                "title": conv.title,
                "totalTokens": bucket["totalTokens"],
                "runs": bucket["runs"],
                "updatedAt": conv.updated_at,
            }
        )

    return {
        "today": today,
        "week": week,
        "allTime": all_time,
        "topConversations": top_conversations,
        "byAgent": [
            {
                "agentId": agent_id,
                "name": agent_names.get(agent_id, agent_id),
                "totalTokens": bucket["totalTokens"],
                "runs": bucket["runs"],
            }
            for agent_id, bucket in sorted(
                by_agent.items(), key=lambda kv: kv[1]["totalTokens"], reverse=True
            )
        ],
        "byModel": [
            {
                "model": model,
                "totalTokens": bucket["totalTokens"],
                "runs": bucket["runs"],
            }
            for model, bucket in sorted(
                by_model.items(), key=lambda kv: kv[1]["totalTokens"], reverse=True
            )
        ],
    }
