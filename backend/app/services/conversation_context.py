"""跨 run 对话历史：把 conversation 的 messages 序列化成 OpenAI ChatMessage 数组，
交给 custom adapter 拼到 [system, ...history, currentUser] 中间，让 agent 记住上下文。

序列化规则（契约，前端行为依赖这套语义）：
- 只取 status='complete' 的消息（streaming / 失败的不进历史）；
- 当前 agent 自己的发言 → assistant；别 agent 的发言（群聊）→ `[名字] ...` 的
  user 消息；thinking / tool_use / tool_result 一律不回放，artifact 只折叠成
  `[产物: 标题 (id=...)]` 占位；
- pinned 消息无视截断永远注入（用户的 pin 是显式契约），且不被 token 预算丢弃；
- 超预算时从老到新丢非 pinned 项。

上下文摘要：摘要的**生成**在后续阶段接入；读取最新摘要与渲染摘要块的辅助
函数在本模块（子 Agent 上下文与 prompt 前缀会用到），summary 为空即无操作。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Agent, Artifact, ContextSummary, Conversation, Message
from app.utils.model_registry import estimate_tokens

DEFAULT_MAX_TURNS = 20


async def build_history_for(
    session: AsyncSession,
    agent_id: str,
    conversation_id: str,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    include_pinned: bool = True,
    exclude_message_id: str | None = None,
    token_budget: int | None = None,
) -> list[dict[str, Any]]:
    """DB 读 + 纯转换。失败由调用方捕获并退化到「无历史」，不影响主流程。"""
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))

    # 最近 N 条 complete 消息（触发消息本身排除，避免重复）
    conditions = [
        Message.conversation_id == conversation_id,
        Message.status == "complete",
    ]
    if exclude_message_id:
        conditions.append(Message.id != exclude_message_id)
    recent_stmt = (
        select(Message)
        .where(*conditions)
        .order_by(Message.created_at.desc())
        .limit(max_turns)
    )
    recent = list(await session.scalars(recent_stmt))[::-1]  # 逆序取的，翻回正序

    # pinned 消息可能在最近 N 条之外，单独拉
    pinned: list[Message] = []
    pinned_ids: set[str] = set()
    if include_pinned and conv is not None:
        wanted = [i for i in (conv.pinned_message_ids or []) if i != exclude_message_id]
        if wanted:
            pinned = list(
                await session.scalars(
                    select(Message).where(
                        Message.id.in_(wanted),
                        Message.status == "complete",
                    )
                )
            )
            pinned_ids = {m.id for m in pinned}

    # 群聊（>1 个 agent）时把别 agent 的发言渲染成 `[名字]: text` 的 user 消息
    agent_names: dict[str, str] = {}
    if conv is not None and len(conv.agent_ids or []) > 1:
        result = await session.execute(
            select(Agent.id, Agent.name).where(Agent.id.in_(conv.agent_ids))
        )
        agent_names = {r[0]: r[1] for r in result.all()}

    # 合并去重，按时间升序
    by_id: dict[str, Message] = {}
    for m in recent:
        by_id[m.id] = m
    for m in pinned:
        by_id[m.id] = m
    merged = sorted(by_id.values(), key=lambda m: m.created_at)

    # artifact_ref 折叠需要标题，批量取一次
    artifact_titles = await _load_artifact_titles(session, _collect_artifact_ids(merged))

    # 先序列化全量，再按 token 预算从老往新丢非 pinned 项
    items: list[dict[str, Any]] = []
    for msg in merged:
        serialized = _serialize_message(msg, agent_id, artifact_titles, agent_names)
        if not serialized:
            continue
        tokens = sum(_estimate_chat_message_tokens(m) for m in serialized)
        items.append(
            {"is_pinned": msg.id in pinned_ids, "serialized": serialized, "tokens": tokens}
        )

    if token_budget is not None and token_budget > 0:
        total = sum(it["tokens"] for it in items)
        for item in items:
            if total <= token_budget:
                break
            if item["is_pinned"]:
                continue
            total -= item["tokens"]
            item["tokens"] = -1  # 标记丢弃

    out: list[dict[str, Any]] = []
    for item in items:
        if item["tokens"] < 0:
            continue
        out.extend(item["serialized"])
    return out


# ─── token 估算（粗粒度，4 字符≈1 token，每条 message 加 4 兜底）──────


def _estimate_chat_message_tokens(m: dict[str, Any]) -> int:
    s = ""
    content = m.get("content")
    if isinstance(content, str):
        s += content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                s += part.get("text", "")
    # 每条 message 至少有 role / metadata 开销
    return estimate_tokens(s) + 4


# ─── 序列化核心 ─────────────────────────────────────────


def _serialize_message(
    msg: Message,
    current_agent_id: str,
    artifact_titles: dict[str, str],
    agent_names: dict[str, str],
) -> list[dict[str, Any]] | None:
    # system prompt 由 runner 注入，不进 history
    if msg.role == "system":
        return None

    if msg.role == "user":
        content = _render_user_parts(msg.parts or [])
        if not content:
            return None
        return [{"role": "user", "content": content}]

    if msg.role == "agent":
        if msg.agent_id == current_agent_id:
            text = _render_agent_public_text(msg.parts or [], artifact_titles)
            if not text:
                return None
            return [{"role": "assistant", "content": text}]
        # 群聊：别 agent 的消息 → `[名字] text` 的 user 消息；单聊不会走到这
        if msg.agent_id and msg.agent_id in agent_names:
            text = _render_agent_public_text(msg.parts or [], artifact_titles)
            if not text:
                return None
            return [{"role": "user", "content": f"[{agent_names[msg.agent_id]}] {text}"}]
        return None

    return None


def _render_user_parts(parts: list[Any]) -> str:
    buf: list[str] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            buf.append(p.get("content", ""))
        elif p.get("type") == "image_attachment":
            buf.append(f"[图片附件: {p.get('fileName')}]")
        elif p.get("type") == "file_attachment":
            buf.append(f"[文件附件: {p.get('fileName')}]")
        # thinking / tool_use 等不应出现在 user 消息里，跳过
    return "\n".join(buf).strip()


def _render_agent_public_text(parts: list[Any], artifact_titles: dict[str, str]) -> str:
    """agent 消息的公开输出：text / code / artifact_ref / deploy_status 折叠；
    thinking / tool_use / tool_result 一律丢——「文字是公共语言，思考与工具调用是私事」。"""
    buf: list[str] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type")
        if ptype == "text":
            if p.get("content"):
                buf.append(p["content"])
        elif ptype == "code":
            if p.get("content"):
                buf.append(p["content"])
        elif ptype == "artifact_ref":
            title = artifact_titles.get(p.get("artifactId", ""), "")
            buf.append(
                f"[产物: {title} (id={p.get('artifactId')})]"
                if title
                else f"[产物 {p.get('artifactId')}]"
            )
        elif ptype == "deploy_status":
            deployment = p.get("deployment") or {}
            if deployment.get("status") == "ready":
                buf.append(
                    f"[部署预览: {deployment.get('title')} "
                    f"{_deployment_source_label(deployment)} ({deployment.get('previewPath')})]"
                )
            else:
                buf.append(
                    f"[部署失败: {deployment.get('title')} "
                    f"({deployment.get('error') or 'unknown error'})]"
                )
    return "\n".join(buf).strip()


def _deployment_source_label(deployment: dict[str, Any]) -> str:
    if deployment.get("sourceType") == "workspace":
        return f"workspace={deployment.get('workspacePath') or 'unknown'}"
    return f"v{deployment.get('version')}"


def _collect_artifact_ids(messages: list[Message]) -> list[str]:
    ids: set[str] = set()
    for m in messages:
        if m.role != "agent":
            continue
        for p in m.parts or []:
            if isinstance(p, dict) and p.get("type") == "artifact_ref":
                ids.add(p.get("artifactId", ""))
    return [i for i in ids if i]


async def _load_artifact_titles(session: AsyncSession, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    result = await session.execute(
        select(Artifact.id, Artifact.title).where(Artifact.id.in_(ids))
    )
    return {r[0]: r[1] for r in result.all()}


# ─── 上下文摘要（读取 + 渲染）────────────────────────────────


async def get_latest_context_summary(
    session: AsyncSession, conversation_id: str
) -> ContextSummary | None:
    """会话最新一份上下文摘要；没有就是 None。"""
    return await session.scalar(
        select(ContextSummary)
        .where(ContextSummary.conversation_id == conversation_id)
        .order_by(ContextSummary.created_at.desc())
        .limit(1)
    )


def render_conversation_summary_block(summary: ContextSummary) -> str:
    """摘要 → `<conversation_summary covered_until_message_id="...">` XML 块。"""
    covered = summary.covered_until_message_id or ""
    covered = covered.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    return "\n".join(
        [
            f'<conversation_summary covered_until_message_id="{covered}">',
            summary.summary,
            "</conversation_summary>",
        ]
    )
