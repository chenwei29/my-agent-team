"""SQLAlchemy 2.0 声明式模型 —— 9 张表。

列名保持 snake_case（表结构以本文件的声明为准）；camelCase 只存在于 API 出参，
由 app/schemas 的 CamelModel alias 机制负责。

删除级联靠 SQLite 的 ON DELETE CASCADE，前提是连接上开了 PRAGMA foreign_keys=ON
（见 app/db/session.py）。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Boolean, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ─── Agents ──────────────────────────────────────────────────
class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    avatar: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    capabilities: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    adapter_name: Mapped[str] = mapped_column(Text, nullable=False)

    model_provider: Mapped[str | None] = mapped_column(Text)
    model_id: Mapped[str | None] = mapped_column(Text)
    api_key: Mapped[str | None] = mapped_column(Text)
    api_base_url: Mapped[str | None] = mapped_column(Text)

    tool_names: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    is_builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_orchestrator: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    supports_vision: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


# ─── Conversations ───────────────────────────────────────────
class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(Text, nullable=False)  # 'single' | 'group'
    agent_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    pinned_message_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    bookmarked_message_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    pinned_at: Mapped[int | None] = mapped_column(Integer)
    fs_write_approval_mode: Mapped[str] = mapped_column(Text, nullable=False, default="review")

    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (Index("idx_conv_updated", "updated_at"),)


# ─── Messages ────────────────────────────────────────────────
class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        Text, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )

    role: Mapped[str] = mapped_column(Text, nullable=False)  # 'user' | 'agent' | 'system'
    agent_id: Mapped[str | None] = mapped_column(Text, ForeignKey("agents.id"))

    parts: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)

    status: Mapped[str] = mapped_column(Text, nullable=False)
    parent_message_id: Mapped[str | None] = mapped_column(Text)
    mentioned_agent_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    run_id: Mapped[str | None] = mapped_column(Text)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    created_at: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (Index("idx_messages_conv_created", "conversation_id", "created_at"),)


# ─── Artifacts ───────────────────────────────────────────────
class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        Text, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )

    type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    parent_artifact_id: Mapped[str | None] = mapped_column(Text)

    created_by_agent_id: Mapped[str] = mapped_column(
        Text, ForeignKey("agents.id"), nullable=False
    )
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (Index("idx_artifacts_conv", "conversation_id"),)


# ─── Workspaces ──────────────────────────────────────────────
class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    root_path: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(Text, nullable=False, default="sandbox")
    bound_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


# ─── Attachments (会话文件库) ─────────────────────────────────
class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        Text, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )

    kind: Mapped[str] = mapped_column(Text, nullable=False)  # 'image' | 'file'
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)  # 相对 workspace.root_path
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (Index("idx_attachments_conv", "conversation_id"),)


# ─── AgentRuns ───────────────────────────────────────────────
class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        Text, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    agent_id: Mapped[str] = mapped_column(Text, ForeignKey("agents.id"), nullable=False)
    trigger_message_id: Mapped[str | None] = mapped_column(Text)

    status: Mapped[str] = mapped_column(Text, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    parent_run_id: Mapped[str | None] = mapped_column(Text)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    started_at: Mapped[int] = mapped_column(Integer, nullable=False)
    finished_at: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (Index("idx_runs_parent", "parent_run_id"),)


# ─── Conversation context summaries ──────────────────────────
class ContextSummary(Base):
    __tablename__ = "conversation_context_summaries"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        Text, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    covered_until_message_id: Mapped[str] = mapped_column(Text, nullable=False)
    covered_until_created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    source_message_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False)
    model_provider: Mapped[str | None] = mapped_column(Text)
    model_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("idx_context_summaries_conv_created", "conversation_id", "created_at"),
    )


# ─── AppSettings (全局 API key / endpoint) ──────────────────
class AppSettings(Base):
    """单行表，PK 固定 'singleton'。"""

    __tablename__ = "app_settings"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    anthropic_api_key: Mapped[str | None] = mapped_column(Text)
    anthropic_base_url: Mapped[str | None] = mapped_column(Text)
    openai_api_key: Mapped[str | None] = mapped_column(Text)
    deepseek_api_key: Mapped[str | None] = mapped_column(Text)
    ark_api_key: Mapped[str | None] = mapped_column(Text)
    deployment_publish_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    deployment_publish_dir: Mapped[str | None] = mapped_column(Text)
    deployment_public_base_url: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)
