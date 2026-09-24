"""上下文压缩：把较早的会话历史交给 LLM 压成一份摘要，落 `conversation_context_summaries`。

压缩后的影响面（读取侧在 conversation_context.build_history_for）：
- 历史注入变成「摘要块 + 摘要覆盖点之后的消息」，被覆盖的旧消息不再进 LLM 上下文；
- 摘要块以 `<conversation_summary covered_until_message_id="...">` 形式出现，
  在 history 里视同 pinned（token 预算永不丢弃）；
- 同时往会话里插一条可见的 system 消息，让用户在时间线上看到压缩发生过。

摘要模型选择顺序：
1. 会话里第一个「配置齐全的 custom agent」（provider + modelId + 对应 key）；
2. 全局 anthropic key（走 Messages API，可配第三方网关 base URL）；
3. 都没有 → 本地启发式兜底摘要（不调 LLM，provider/modelId 记 null）。

压缩会跳过 pinned 与 system 消息，并保留最近若干条不压缩（RECENT_MESSAGES_TO_KEEP），
让最新上下文保持原文细节。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.custom_provider_client import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_VOLCANO_ARK_BASE_URL,
)
from app.db.models import Agent, Artifact, ContextSummary, Conversation, Message
from app.errors import ServiceError
from app.services.conversation_context import (
    get_latest_context_summary,
    render_conversation_summary_block,
)
from app.services.settings_service import (
    get_effective_anthropic_base_url,
    get_effective_api_key,
)
from app.utils.ids import new_context_summary_id, new_message_id
from app.utils.model_registry import estimate_tokens
from app.utils.time import now_ms

RECENT_MESSAGES_TO_KEEP = 6
MAX_RENDERED_MESSAGE_CHARS = 4000
MAX_COMPACTION_INPUT_CHARS = 60000
SUMMARY_MAX_TOKENS = 1600

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"

COMPACTION_SYSTEM_PROMPT = "\n".join(
    [
        "你是 AgentHub 的上下文压缩器。你的任务是把较早的会话历史压缩成一份可继续工作的摘要。",
        "",
        "输出要求：",
        "- 使用中文，除非历史内容主要是英文。",
        "- 保留用户目标、已经确认的决策、关键约束、未完成问题、重要文件/模块/命令、artifact id、agent 分工和结论。",
        "- 保留会影响后续执行的错误、回滚、测试结果、环境信息。",
        "- 删除寒暄、重复内容、详细工具日志、长代码正文；如果代码很重要，只记录文件、函数、意图和关键差异。",
        "- 不要虚构没有出现过的事实。",
        "- 用简洁分节或项目符号输出，适合直接放进下一轮 LLM 上下文。",
    ]
)


@dataclass
class SummaryModelChoice:
    provider: str | None
    model_id: str | None
    summarize: Callable[[str], Awaitable[str]]


async def prefix_prompt_with_context_summary(
    session: AsyncSession, conversation_id: str, prompt: str
) -> str:
    """给 prompt 前面拼上最新摘要块（没有摘要就原样返回）。

    给不消费 history 数组的 adapter 用（自定义 adapter 走 build_history_for 注入）。
    """
    latest = await get_latest_context_summary(session, conversation_id)
    if latest is None:
        return prompt
    return "\n".join([render_conversation_summary_block(latest), "", prompt])


async def compact_conversation(session: AsyncSession, conversation_id: str) -> dict[str, Any]:
    """压缩会话历史 → 落摘要行 + 插 system 消息 + 刷新会话时间线。"""
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if conv is None:
        raise ServiceError(f"Conversation not found: {conversation_id}")

    latest = await get_latest_context_summary(session, conversation_id)
    messages = await _load_compactable_messages(session, conversation_id, latest)
    pinned_ids = set(conv.pinned_message_ids or [])
    compactable = [m for m in messages if m.id not in pinned_ids and m.role != "system"]
    keep_recent = RECENT_MESSAGES_TO_KEEP if len(compactable) > RECENT_MESSAGES_TO_KEEP else 0
    source = compactable[: max(0, len(compactable) - keep_recent)]

    if not source:
        raise ServiceError("No compactable history yet")

    rendered = await _render_messages_for_compaction(session, source, conv.agent_ids or [])
    if not rendered["text"].strip() or not rendered["included"]:
        raise ServiceError("No compactable text in selected history")

    covered_until = rendered["included"][-1]
    prompt = _build_compaction_prompt(latest, rendered["text"])
    choice = await _choose_summary_model(session, conv.agent_ids or [])
    summary_text = _normalize_summary(await choice.summarize(prompt))
    now = now_ms()

    summary = ContextSummary(
        id=new_context_summary_id(),
        conversation_id=conversation_id,
        summary=summary_text,
        covered_until_message_id=covered_until.id,
        covered_until_created_at=covered_until.created_at,
        source_message_count=len(rendered["included"]),
        token_estimate=estimate_tokens(summary_text),
        model_provider=choice.provider,
        model_id=choice.model_id,
        created_at=now,
    )
    session.add(summary)

    system_message = Message(
        id=new_message_id(),
        conversation_id=conversation_id,
        role="system",
        agent_id=None,
        parts=[
            {
                "type": "text",
                "content": f"已压缩早期上下文，覆盖 {len(rendered['included'])} 条消息。",
            }
        ],
        status="complete",
        parent_message_id=None,
        mentioned_agent_ids=[],
        run_id=None,
        usage=None,
        created_at=now,
    )
    session.add(system_message)

    conv.updated_at = now
    await session.commit()
    return {"summary": summary, "message": system_message}


# ─── 历史加载与渲染 ─────────────────────────────────────────


async def _load_compactable_messages(
    session: AsyncSession,
    conversation_id: str,
    latest: ContextSummary | None,
) -> list[Message]:
    conditions = [Message.conversation_id == conversation_id, Message.status == "complete"]
    if latest is not None:
        conditions.append(Message.created_at > latest.covered_until_created_at)
    stmt = select(Message).where(*conditions).order_by(Message.created_at)
    return list(await session.scalars(stmt))


async def _render_messages_for_compaction(
    session: AsyncSession,
    messages: list[Message],
    conversation_agent_ids: list[str],
) -> dict[str, Any]:
    agent_ids: set[str] = {m.agent_id for m in messages if m.agent_id}
    agent_ids.update(conversation_agent_ids)
    agent_name_by_id: dict[str, str] = {}
    if agent_ids:
        rows = await session.execute(select(Agent.id, Agent.name).where(Agent.id.in_(agent_ids)))
        agent_name_by_id = {r[0]: r[1] for r in rows.all()}

    artifact_ids = _collect_artifact_ids(messages)
    artifact_titles = await _load_artifact_titles(session, artifact_ids)

    chunks: list[str] = []
    included: list[Message] = []
    total_chars = 0
    for message in messages:
        rendered = _render_message_for_compaction(message, agent_name_by_id, artifact_titles)
        if not rendered:
            continue
        chunk = _limit_chars(rendered, MAX_RENDERED_MESSAGE_CHARS)
        if total_chars + len(chunk) > MAX_COMPACTION_INPUT_CHARS:
            break
        chunks.append(chunk)
        included.append(message)
        total_chars += len(chunk)
    return {"text": "\n\n".join(chunks), "included": included}


def _render_message_for_compaction(
    message: Message,
    agent_name_by_id: dict[str, str],
    artifact_titles: dict[str, str],
) -> str | None:
    if message.role == "user":
        from_ = "user"
    elif message.agent_id:
        from_ = agent_name_by_id.get(message.agent_id, message.agent_id)
    else:
        from_ = message.role
    content = _render_public_parts(message.parts or [], artifact_titles)
    if not content.strip():
        return None
    return "\n".join(
        [
            f'<message id="{_escape_attr(message.id)}" '
            f'from="{_escape_attr(from_)}" created_at="{message.created_at}">',
            content,
            "</message>",
        ]
    )


def _render_public_parts(parts: list[Any], artifact_titles: dict[str, str]) -> str:
    out: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            if part.get("content"):
                out.append(part["content"])
        elif part_type == "code":
            if part.get("content"):
                out.append("\n".join(["```" + (part.get("language") or ""), part["content"], "```"]))
        elif part_type == "artifact_ref":
            artifact_id = part.get("artifactId", "")
            title = artifact_titles.get(artifact_id)
            out.append(
                f"[artifact: {title} (id={artifact_id})]" if title else f"[artifact id={artifact_id}]"
            )
        elif part_type == "deploy_status":
            deployment = part.get("deployment") or {}
            if deployment.get("status") == "ready":
                out.append(
                    f"[deployment: {deployment.get('title')} "
                    f"{_deployment_source_label(deployment)} ({deployment.get('previewPath')})]"
                )
            else:
                out.append(
                    f"[deployment failed: {deployment.get('title')} "
                    f"({deployment.get('error') or 'unknown error'})]"
                )
        elif part_type == "image_attachment":
            out.append(f"[image attachment: {part.get('fileName')}, id={part.get('attachmentId')}]")
        elif part_type == "file_attachment":
            out.append(f"[file attachment: {part.get('fileName')}, id={part.get('attachmentId')}]")
    return "\n".join(out).strip()


def _deployment_source_label(deployment: dict[str, Any]) -> str:
    if deployment.get("sourceType") == "workspace":
        return f"workspace={deployment.get('workspacePath') or 'unknown'}"
    return f"v{deployment.get('version')}"


def _collect_artifact_ids(messages: list[Message]) -> list[str]:
    ids: set[str] = set()
    for message in messages:
        for part in message.parts or []:
            if isinstance(part, dict) and part.get("type") == "artifact_ref":
                ids.add(part.get("artifactId", ""))
    return [i for i in ids if i]


async def _load_artifact_titles(session: AsyncSession, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    rows = await session.execute(select(Artifact.id, Artifact.title).where(Artifact.id.in_(ids)))
    return {r[0]: r[1] for r in rows.all()}


def _build_compaction_prompt(latest: ContextSummary | None, rendered_messages: str) -> str:
    previous = (
        "\n".join(["<previous_summary>", latest.summary, "</previous_summary>", ""])
        if latest is not None
        else ""
    )
    return "\n".join([previous, "<messages_to_compact>", rendered_messages, "</messages_to_compact>"])


# ─── 摘要模型选择与调用 ─────────────────────────────────────


async def _choose_summary_model(
    session: AsyncSession, agent_ids: list[str]
) -> SummaryModelChoice:
    agents_by_id: dict[str, Agent] = {}
    if agent_ids:
        rows = await session.scalars(select(Agent).where(Agent.id.in_(agent_ids)))
        agents_by_id = {a.id: a for a in rows}

    for agent_id in agent_ids:
        choice = await _choice_from_custom_agent(session, agents_by_id.get(agent_id))
        if choice is not None:
            return choice

    anthropic_key = await get_effective_api_key(session, "anthropic")
    if anthropic_key:
        claude_agent = next(
            (a for a in agents_by_id.values() if a.adapter_name == "claude-code"), None
        )
        return _build_anthropic_choice(
            anthropic_key,
            await get_effective_anthropic_base_url(session),
            (claude_agent.model_id if claude_agent and claude_agent.model_id else DEFAULT_CLAUDE_MODEL),
        )

    return SummaryModelChoice(provider=None, model_id=None, summarize=_heuristic_summary)


async def _choice_from_custom_agent(
    session: AsyncSession, agent: Agent | None
) -> SummaryModelChoice | None:
    if agent is None or agent.adapter_name != "custom" or not agent.model_provider or not agent.model_id:
        return None

    provider = agent.model_provider
    if provider == "anthropic":
        key = agent.api_key or await get_effective_api_key(session, "anthropic")
        if not key:
            return None
        return _build_anthropic_choice(
            key,
            agent.api_base_url or await get_effective_anthropic_base_url(session),
            agent.model_id,
        )

    if provider == "openai-compatible":
        # key 与 endpoint 成对配置，没有全局兜底
        if not agent.api_key or not agent.api_base_url:
            return None
        return _build_openai_compatible_choice(
            provider, agent.model_id, agent.api_key, agent.api_base_url
        )

    key = agent.api_key or await get_effective_api_key(
        session, "ark" if provider == "volcano-ark" else provider
    )
    if not key:
        return None
    return _build_openai_compatible_choice(provider, agent.model_id, key, agent.api_base_url)


def _build_openai_compatible_choice(
    provider: str,
    model_id: str,
    api_key: str,
    api_base_url: str | None,
) -> SummaryModelChoice:
    if api_base_url:
        base_url: str | None = api_base_url
    elif provider == "deepseek":
        base_url = DEFAULT_DEEPSEEK_BASE_URL
    elif provider == "volcano-ark":
        base_url = DEFAULT_VOLCANO_ARK_BASE_URL
    else:
        base_url = None

    async def summarize(prompt: str) -> str:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=2)
        result = await client.chat.completions.create(
            model=model_id,
            temperature=0.2,
            messages=[
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return result.choices[0].message.content or "" if result.choices else ""

    return SummaryModelChoice(provider=provider, model_id=model_id, summarize=summarize)


def _build_anthropic_choice(
    api_key: str,
    api_base_url: str | None,
    model_id: str,
) -> SummaryModelChoice:
    async def summarize(prompt: str) -> str:
        base = (api_base_url or "https://api.anthropic.com").rstrip("/")
        headers = {"anthropic-version": "2023-06-01"}
        if api_base_url:
            # 自定义网关按 Bearer 传 key
            headers["authorization"] = f"Bearer {api_key}"
        else:
            headers["x-api-key"] = api_key
        transport = httpx.AsyncHTTPTransport(retries=2)
        async with httpx.AsyncClient(transport=transport, timeout=120) as client:
            response = await client.post(
                f"{base}/v1/messages",
                headers=headers,
                json={
                    "model": model_id,
                    "max_tokens": SUMMARY_MAX_TOKENS,
                    "temperature": 0.2,
                    "system": COMPACTION_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            blocks = response.json().get("content") or []
        return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    return SummaryModelChoice(provider="anthropic", model_id=model_id, summarize=summarize)


async def _heuristic_summary(prompt: str) -> str:
    return "\n".join(
        [
            "本摘要由本地兜底规则生成，因为当前没有可用的摘要模型 API key。",
            "",
            "已压缩的早期上下文如下。后续模型应把它视为旧对话摘要，并优先结合最新未压缩消息判断用户意图。",
            "",
            _limit_chars(prompt, 10000),
        ]
    )


def _normalize_summary(summary: str) -> str:
    trimmed = summary.strip()
    if not trimmed:
        raise ServiceError("Compaction model returned an empty summary")
    return trimmed


def _limit_chars(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[truncated]"


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
