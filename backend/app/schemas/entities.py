"""实体出入参模型：Agent / Conversation / Message / Workspace / Settings 的响应与请求体。

字段形状以 web/src/shared/types.ts 为准；请求体的约束（min/max、enum、严格模式、
跨字段校验）也在这里一次性写死 —— 改约束等于改 API 契约。

⚠️ 事件模型不走 CamelModel 的 alias 机制（字段名本身就手写 camelCase），见 schemas/events.py。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from app.schemas.base import CamelModel

# ─── 枚举（与 web/src/shared/types.ts 对齐）──────────────────
AdapterName = Literal["custom", "claude-code", "codex"]
ModelProvider = Literal["anthropic", "openai", "deepseek", "volcano-ark", "openai-compatible"]
ConversationMode = Literal["single", "group"]
FsWriteApprovalMode = Literal["auto", "review"]
ServerPlatform = Literal["posix", "windows"]

SINGLETON_ID = "singleton"


# ─── 响应模型 ────────────────────────────────────────────────
class AgentOut(CamelModel):
    id: str
    name: str
    avatar: str
    description: str
    capabilities: list[str]
    system_prompt: str
    adapter_name: str
    model_provider: str | None = None
    model_id: str | None = None
    api_key: str | None = None
    api_base_url: str | None = None
    tool_names: list[str]
    is_builtin: bool
    is_orchestrator: bool
    supports_vision: bool
    created_at: int


class ConversationOut(CamelModel):
    id: str
    title: str
    mode: str
    agent_ids: list[str]
    pinned_message_ids: list[str]
    bookmarked_message_ids: list[str]
    archived: bool
    pinned_at: int | None = None
    fs_write_approval_mode: str
    created_at: int
    updated_at: int


class ConversationWithMetaOut(ConversationOut):
    """会话行 + workspace 的 mode / boundPath —— 前端多处显示「本地工作目录」标识。"""

    workspace_mode: str
    workspace_bound_path: str | None = None


class WorkspaceOut(CamelModel):
    id: str
    conversation_id: str
    root_path: str
    mode: str
    bound_path: str | None = None
    created_at: int


class MessageOut(CamelModel):
    id: str
    conversation_id: str
    role: str
    agent_id: str | None = None
    # parts 走原样透传：P1 只产出 text / *_attachment，判别联合留到 P3 adapter 开始产 part 时再收
    parts: list[dict[str, Any]]
    status: str
    parent_message_id: str | None = None
    mentioned_agent_ids: list[str]
    run_id: str | None = None
    usage: dict[str, Any] | None = None
    created_at: int


class AppSettingsOut(CamelModel):
    id: str
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    ark_api_key: str | None = None
    deployment_publish_enabled: bool
    deployment_publish_dir: str | None = None
    deployment_public_base_url: str | None = None
    updated_at: int


# ─── 响应信封（前端读的是 { agents: [...] } / { conversation: {...} } 这类包一层）──
class AgentsResponse(CamelModel):
    agents: list[AgentOut]


class AgentResponse(CamelModel):
    agent: AgentOut


class ConversationsResponse(CamelModel):
    conversations: list[ConversationWithMetaOut]


class ConversationResponse(CamelModel):
    conversation: ConversationWithMetaOut


class MessagesResponse(CamelModel):
    messages: list[MessageOut]


class SettingsResponse(CamelModel):
    settings: AppSettingsOut


class PlatformResponse(CamelModel):
    platform: ServerPlatform


class OkResponse(CamelModel):
    ok: bool


class SendMessageResponse(CamelModel):
    message_id: str
    run_ids: list[str]
    messages: list[MessageOut] | None = None
    # 内容形如 /deploy 的指令消息不发 run，而是当场部署并把结果挂在这里
    deploy: dict[str, Any] | None = None


class DeployCandidatesResponse(CamelModel):
    candidates: list[dict[str, Any]]


class ClearHistoryResponse(CamelModel):
    conversation: ConversationWithMetaOut
    deleted_message_count: int
    deleted_run_count: int
    deleted_summary_count: int


class ListDirEntry(CamelModel):
    name: str
    is_directory: bool
    path: str | None = None


class ListDirResponse(CamelModel):
    path: str
    parent: str | None = None
    entries: list[ListDirEntry]


# ─── 请求体：Agents ─────────────────────────────────────────
# 前端一律发 camelCase，所以请求体也用 CamelModel 的 alias 机制（populate_by_name 让 Python
# 侧仍可用 snake_case 构造），否则 systemPrompt / modelProvider 这类字段根本收不到。
class CreateAgentBody(CamelModel):
    name: Annotated[str, Field(min_length=1, max_length=64)]
    avatar: Annotated[str, Field(max_length=8)] = ""
    description: Annotated[str, Field(min_length=1, max_length=280)]
    capabilities: list[str] = Field(default_factory=list)
    system_prompt: Annotated[str, Field(min_length=1)]
    adapter_name: AdapterName = "custom"
    model_provider: ModelProvider | None = None
    model_id: Annotated[str, Field(min_length=1)] | None = None
    tool_names: list[str] = Field(default_factory=list)
    supports_vision: bool | None = None
    api_key: str | None = None
    api_base_url: str | None = None

    @model_validator(mode="after")
    def _custom_requires_provider_and_model(self) -> CreateAgentBody:
        if self.adapter_name == "custom" and (not self.model_provider or not self.model_id):
            raise ValueError("Custom adapter requires modelProvider and modelId")
        return self


class UpdateAgentBody(CamelModel):
    """`extra='forbid'`：多余字段直接 400（严格模式）。"""

    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, from_attributes=True, extra="forbid"
    )

    name: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    description: Annotated[str, Field(min_length=1, max_length=280)] | None = None
    capabilities: list[str] | None = None
    system_prompt: Annotated[str, Field(min_length=1)] | None = None
    adapter_name: AdapterName | None = None
    model_provider: ModelProvider | None = None
    model_id: Annotated[str, Field(min_length=1)] | None = None
    tool_names: list[str] | None = None
    supports_vision: bool | None = None
    api_key: str | None = None
    api_base_url: str | None = None


# ─── 请求体：Conversations ──────────────────────────────────
class CreateConversationBody(CamelModel):
    title: str | None = None
    mode: ConversationMode
    agent_ids: Annotated[list[str], Field(min_length=1)]
    bound_path: str | None = None


class PatchConversationBody(CamelModel):
    """同一端点 5 种语义，按 body 字段区分（前端调用处见 web/src/lib/api.ts）。

    togglePin / toggleArchive 只接受 true —— 传 false 视为非法请求体，不是「取消勾选」。
    """

    add_agent_ids: Annotated[list[str], Field(min_length=1)] | None = None
    title: Annotated[str, Field(min_length=1, max_length=100)] | None = None
    fs_write_approval_mode: FsWriteApprovalMode | None = None
    toggle_pin: Literal[True] | None = None
    toggle_archive: Literal[True] | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> PatchConversationBody:
        if (
            self.add_agent_ids is None
            and self.title is None
            and self.fs_write_approval_mode is None
            and self.toggle_pin is None
            and self.toggle_archive is None
        ):
            raise ValueError(
                "At least one of addAgentIds / title / fsWriteApprovalMode / togglePin / "
                "toggleArchive is required"
            )
        return self


class SendMessageBody(CamelModel):
    content: str = ""
    mentioned_agent_ids: list[str] | None = None
    parent_message_id: str | None = None
    attachment_ids: list[str] | None = None

    @model_validator(mode="after")
    def _content_or_attachments(self) -> SendMessageBody:
        if not self.content.strip() and not self.attachment_ids:
            raise ValueError("必须提供 content 或 attachmentIds 之一")
        return self


# ─── 请求体：部署 ──────────────────────────────────────────
class DeployConversationBody(CamelModel):
    """空对象也是合法请求体（走自动判定候选）。"""

    artifact_id: Annotated[str, Field(min_length=1)] | None = None


# ─── 请求体：消息高级操作（撤回 / 编辑 / pin / bookmark）─────
class WithdrawMessageBody(CamelModel):
    conversation_id: Annotated[str, Field(min_length=1)]


class EditMessageBody(CamelModel):
    conversation_id: Annotated[str, Field(min_length=1)]
    content: Annotated[str, Field(min_length=1)]


# ─── 请求体：Settings ───────────────────────────────────────
class UpdateSettingsBody(CamelModel):
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    ark_api_key: str | None = None
    deployment_publish_enabled: bool | None = None
    deployment_publish_dir: str | None = None
    deployment_public_base_url: str | None = None


# ─── 会话内文件（FileTab 手动浏览，与工具沙箱共用 fs_service） ──
class WorkspaceFsEntry(CamelModel):
    name: str
    is_directory: bool
    size: int | None = None


class WorkspaceListResult(CamelModel):
    rel_path: str
    absolute_path: str
    parent: str | None
    entries: list[WorkspaceFsEntry]


class WorkspaceReadResult(CamelModel):
    path: str
    absolute_path: str
    cwd: str
    size: int
    content: str
    truncated: bool


class WorkspaceWriteResult(CamelModel):
    path: str
    absolute_path: str
    cwd: str
    bytes: int


class WorkspaceFsWriteBody(CamelModel):
    path: Annotated[str, Field(min_length=1)]
    content: str


# ─── 审批请求体 ─────────────────────────────────────────────
class ResolvePendingBody(CamelModel):
    action: Literal["approve", "reject"]


class AskUserAnswerBody(CamelModel):
    selected_labels: list[str]
    freeform_note: str | None = None


class AnswerQuestionsBody(CamelModel):
    answers: dict[str, AskUserAnswerBody]


class ReviewDispatchPlanBody(CamelModel):
    """待审分派计划的审批请求体：approve/reject 不带额外字段；revise 必带 feedback。"""

    action: Literal["approve", "reject", "revise"]
    feedback: str | None = None

    @model_validator(mode="after")
    def _revise_needs_feedback(self) -> ReviewDispatchPlanBody:
        if self.action == "revise" and (
            self.feedback is None or not (1 <= len(self.feedback) <= 4000)
        ):
            raise ValueError("revise 需要 1-4000 字符的 feedback")
        return self


# ─── 附件 ──────────────────────────────────────────────────
class AttachmentOut(CamelModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    kind: str
    file_name: str
    file_path: str
    size: int
    mime_type: str
    created_at: int


class AttachmentResponse(CamelModel):
    attachment: AttachmentOut


class AttachmentsResponse(CamelModel):
    attachments: list[AttachmentOut]

