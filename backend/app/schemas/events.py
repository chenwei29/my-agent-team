"""SSE 事件流（StreamEvent）的判别联合 —— 与前端 `web/src/shared/types.ts` 的 StreamEvent 联合逐字对齐。

约束：
- 字段名**手写 camelCase**，不走 CamelModel 的 to_camel alias —— 事件是直传 JSON，
  多一层 alias 转换就多一类「字段静默变 snake_case」的 bug。
- 判别字段统一 `type`，Frontend 按 `(messageId, partIndex)` 定位 part。
- `connected` / `heartbeat` **不在**前端的 StreamEvent 联合里（它们由 SSE 层直接发裸对象），
  但两者都带 `timestamp`，所以这里单独建模。

P4–P6 才用到的 payload（PendingWrite / PendingBashCommand / PendingQuestion /
PendingDispatchPlan / DeployStatusRecord …）这里先把**判别字段与字段名**定好，
内部结构暂时用 dict 透传，到对应阶段再收紧类型 —— 这样联合本身以后不用改。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

RunStatus = Literal["complete", "failed", "aborted"]
MessageRole = Literal["user", "agent", "system"]
MessageStatus = Literal["streaming", "complete", "error", "aborted"]


# ─── MessagePart 判别联合（文本 / 代码 / 思考 / 工具调用与结果 / 产物引用 / 附件等 part）──
class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"] = "text"
    content: str


class CodePart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["code"] = "code"
    language: str
    content: str


class ThinkingPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["thinking"] = "thinking"
    content: str


class ToolUsePart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_use"] = "tool_use"
    callId: str
    toolName: str
    args: Any = None


class ToolResultPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_result"] = "tool_result"
    callId: str
    result: Any = None
    isError: bool


class ArtifactRefPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["artifact_ref"] = "artifact_ref"
    artifactId: str


class DeployStatusPart(BaseModel):
    """`deployment` 的完整结构属 P5，先透传。"""

    model_config = ConfigDict(extra="forbid")
    type: Literal["deploy_status"] = "deploy_status"
    deployment: dict[str, Any]


class DeployCandidatesPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["deploy_candidates"] = "deploy_candidates"
    candidates: list[dict[str, Any]]


class ImageAttachmentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image_attachment"] = "image_attachment"
    attachmentId: str
    fileName: str
    size: int
    mimeType: str


class FileAttachmentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["file_attachment"] = "file_attachment"
    attachmentId: str
    fileName: str
    size: int
    mimeType: str


MessagePart = Annotated[
    Union[
        TextPart,
        CodePart,
        ThinkingPart,
        ToolUsePart,
        ToolResultPart,
        ArtifactRefPart,
        DeployStatusPart,
        DeployCandidatesPart,
        ImageAttachmentPart,
        FileAttachmentPart,
    ],
    Field(discriminator="type"),
]


# ─── PartDelta：part 的增量更新，只有 text / code / thinking 3 种，增量字段一律叫 text ──
class TextAppendDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text.append"] = "text.append"
    text: str


class CodeAppendDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["code.append"] = "code.append"
    text: str


class ThinkingAppendDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["thinking.append"] = "thinking.append"
    text: str


PartDelta = Annotated[
    Union[TextAppendDelta, CodeAppendDelta, ThinkingAppendDelta],
    Field(discriminator="type"),
]


# ─── usage payload：run.usage / message.usage 携带的 token 统计 ──────────────
class RunUsageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputTokens: int
    outputTokens: int
    cacheCreationTokens: int
    cacheReadTokens: int
    lastInputTokens: int | None = None
    model: str | None = None


class MessageUsageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputTokens: int
    outputTokens: int
    cacheReadTokens: int


# ─── 记录型 payload ────────────────────────────────────────
class MessageRecord(BaseModel):
    """message.added 事件里整条消息的形状（字段名即 camelCase）。

    刻意不复用 P1 的 `entities.MessageOut`：那个走 CamelModel alias，
    事件里要的是「字段名本身就是 camelCase」，混用两套机制最容易出错。
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    conversationId: str
    role: MessageRole
    agentId: str | None = None
    parts: list[MessagePart]
    status: MessageStatus
    parentMessageId: str | None = None
    mentionedAgentIds: list[str]
    runId: str | None = None
    usage: MessageUsageEvent | None = None
    createdAt: int


class ArtifactRecord(BaseModel):
    """产物记录（artifact.create 事件的 artifact 字段）；`content` 的判别联合属 P5，先透传。"""

    model_config = ConfigDict(extra="forbid")
    id: str
    conversationId: str
    type: str
    title: str
    content: dict[str, Any]
    version: int
    parentArtifactId: str | None = None
    createdByAgentId: str
    createdAt: int


class PendingWrite(BaseModel):
    """等待用户确认的文件写入（fs_write.pending 的 payload）；P4 落地时再收紧字段类型。"""

    model_config = ConfigDict(extra="forbid")
    id: str
    conversationId: str
    agentId: str
    runId: str
    path: str
    absolutePath: str
    oldContent: str | None = None
    newContent: str
    createdAt: int


class PendingBashCommand(BaseModel):
    """等待用户确认的 bash 命令（bash_command.pending 的 payload）；P4 落地时再收紧字段类型。"""

    model_config = ConfigDict(extra="forbid")
    id: str
    conversationId: str
    agentId: str
    runId: str
    command: str
    cwd: str
    reason: str
    createdAt: int


class PendingQuestion(BaseModel):
    """等待用户回答的一组问题（ask_user.pending 的 payload）；questions 的元素结构 P4 再定。"""

    model_config = ConfigDict(extra="forbid")
    id: str
    conversationId: str
    agentId: str
    runId: str
    questions: list[dict[str, Any]]
    createdAt: int


class DispatchPlanItem(BaseModel):
    """分派计划里的一条子任务：负责的 agent、任务描述、依赖、期望产出与验收标准；P6 落地，若干枚举字段先用 str。"""

    model_config = ConfigDict(extra="forbid")
    id: str
    agentId: str
    task: str
    taskKind: str | None = None
    dependsOn: list[str] | None = None
    expectedOutputs: list[str] | None = None
    inputs: list[dict[str, Any]] | None = None
    acceptanceCriteria: list[str] | None = None
    targetPaths: list[str] | None = None
    expectedWorkspaceChanges: list[str] | None = None
    requiredCommands: list[dict[str, Any]] | None = None
    requiredEvidence: list[str] | None = None


class PendingDispatchPlan(BaseModel):
    """等待用户确认的整份分派计划（dispatch.plan.pending 的 payload）；P6 落地。"""

    model_config = ConfigDict(extra="forbid")
    id: str
    conversationId: str
    agentId: str
    runId: str
    plan: list[DispatchPlanItem]
    createdAt: int


# ─── 事件基类：除 connected / heartbeat 外都带这两个字段 ──────
class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversationId: str
    timestamp: int


class RunStartEvent(_Event):
    type: Literal["run.start"] = "run.start"
    runId: str
    agentId: str
    triggerMessageId: str
    parentRunId: str | None = None


class RunEndEvent(_Event):
    type: Literal["run.end"] = "run.end"
    runId: str
    status: RunStatus
    error: str | None = None


class RunUsageEventWrapper(_Event):
    type: Literal["run.usage"] = "run.usage"
    runId: str
    usage: RunUsageEvent


class MessageStartEvent(_Event):
    type: Literal["message.start"] = "message.start"
    messageId: str
    agentId: str
    runId: str


class MessageEndEvent(_Event):
    type: Literal["message.end"] = "message.end"
    messageId: str


class MessageUsageEventWrapper(_Event):
    type: Literal["message.usage"] = "message.usage"
    messageId: str
    usage: MessageUsageEvent


class MessageAddedEvent(_Event):
    type: Literal["message.added"] = "message.added"
    message: MessageRecord


class MessageRemovedEvent(_Event):
    type: Literal["message.removed"] = "message.removed"
    messageIds: list[str]
    artifactIds: list[str]


class PartStartEvent(_Event):
    type: Literal["part.start"] = "part.start"
    messageId: str
    partIndex: int
    part: MessagePart


class PartDeltaEvent(_Event):
    type: Literal["part.delta"] = "part.delta"
    messageId: str
    partIndex: int
    delta: PartDelta


class PartEndEvent(_Event):
    type: Literal["part.end"] = "part.end"
    messageId: str
    partIndex: int


class ToolCallEvent(_Event):
    type: Literal["tool.call"] = "tool.call"
    messageId: str
    callId: str
    toolName: str
    args: Any = None


class ToolResultEvent(_Event):
    type: Literal["tool.result"] = "tool.result"
    messageId: str
    callId: str
    result: Any = None
    isError: bool


class ArtifactCreateEvent(_Event):
    type: Literal["artifact.create"] = "artifact.create"
    artifact: ArtifactRecord


class ArtifactUpdateEvent(_Event):
    type: Literal["artifact.update"] = "artifact.update"
    artifactId: str
    patch: dict[str, Any]


class DeployStatusEvent(_Event):
    type: Literal["deploy.status"] = "deploy.status"
    messageId: str
    deployment: dict[str, Any]


class DispatchPlanPendingEvent(_Event):
    type: Literal["dispatch.plan.pending"] = "dispatch.plan.pending"
    pendingPlan: PendingDispatchPlan


class DispatchPlanResolvedEvent(_Event):
    type: Literal["dispatch.plan.resolved"] = "dispatch.plan.resolved"
    pendingId: str
    runId: str
    approved: bool
    revising: bool | None = None


class DispatchPlanEvent(_Event):
    type: Literal["dispatch.plan"] = "dispatch.plan"
    runId: str
    plan: list[DispatchPlanItem]


class DispatchStartEvent(_Event):
    type: Literal["dispatch.start"] = "dispatch.start"
    parentRunId: str
    childRunId: str
    taskId: str
    agentId: str


class DispatchEndEvent(_Event):
    type: Literal["dispatch.end"] = "dispatch.end"
    parentRunId: str
    taskId: str
    status: str
    childRunId: str | None = None
    error: str | None = None


class FsWritePendingEvent(_Event):
    type: Literal["fs_write.pending"] = "fs_write.pending"
    pendingWrite: PendingWrite


class FsWriteResolvedEvent(_Event):
    type: Literal["fs_write.resolved"] = "fs_write.resolved"
    pendingId: str
    applied: bool


class BashCommandPendingEvent(_Event):
    type: Literal["bash_command.pending"] = "bash_command.pending"
    pendingCommand: PendingBashCommand


class BashCommandResolvedEvent(_Event):
    type: Literal["bash_command.resolved"] = "bash_command.resolved"
    pendingId: str
    approved: bool


class AskUserPendingEvent(_Event):
    type: Literal["ask_user.pending"] = "ask_user.pending"
    pendingQuestion: PendingQuestion


class AskUserResolvedEvent(_Event):
    type: Literal["ask_user.resolved"] = "ask_user.resolved"
    pendingId: str
    answered: bool


class HeartbeatEvent(_Event):
    """心跳事件：SSE 实际只发 `{type, timestamp}` —— 没有 conversationId。

    前端的 applyEvent 对 heartbeat 直接 return，不看 timestamp，
    但这个字段**必须带**，所以 conversationId 设为可选。
    """

    conversationId: str | None = None
    type: Literal["heartbeat"] = "heartbeat"


StreamEvent = Annotated[
    Union[
        RunStartEvent,
        RunEndEvent,
        RunUsageEventWrapper,
        MessageStartEvent,
        MessageEndEvent,
        MessageUsageEventWrapper,
        MessageAddedEvent,
        MessageRemovedEvent,
        PartStartEvent,
        PartDeltaEvent,
        PartEndEvent,
        ToolCallEvent,
        ToolResultEvent,
        ArtifactCreateEvent,
        ArtifactUpdateEvent,
        DeployStatusEvent,
        DispatchPlanPendingEvent,
        DispatchPlanResolvedEvent,
        DispatchPlanEvent,
        DispatchStartEvent,
        DispatchEndEvent,
        FsWritePendingEvent,
        FsWriteResolvedEvent,
        BashCommandPendingEvent,
        BashCommandResolvedEvent,
        AskUserPendingEvent,
        AskUserResolvedEvent,
        HeartbeatEvent,
    ],
    Field(discriminator="type"),
]


class ConnectedEvent(BaseModel):
    """SSE 建连后的第一条消息：裸对象，不属于 StreamEvent 联合。"""

    model_config = ConfigDict(extra="forbid")
    type: Literal["connected"] = "connected"
    timestamp: int


def parse_event(raw: Any) -> StreamEvent:
    """校验一个事件的 JSON 形状（测试与调试用）。"""
    from pydantic import TypeAdapter

    return TypeAdapter(StreamEvent).validate_python(raw)
