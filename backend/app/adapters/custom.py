"""CustomAgentAdapter —— 自配置 Agent 的适配器。

通过 openai SDK 调用底层模型（OpenAI Chat Completions 兼容协议），自己实现
tool loop：流式拉模型输出 → 解析 tool_calls → 执行工具 → 把结果回灌到
messages → 续写下一轮，直到模型不再调工具或达到 MAX_TURNS。

流式 chunk → StreamEvent 的映射：
- delta.content                → part.delta{text.append}
- delta.reasoning_content      → part.delta{thinking.append}（thinking 模型）
- delta.tool_calls             → 按 index 累积参数分片，整段结束后发 tool.call
- chunk.usage（stream_options）→ message.usage / run.usage

中止语义：signal 只在循环头轮询（已建立的 HTTP 流在 return / 异常时由
async with 确保关闭）；task 被 cancel 时 CancelledError 穿透、连接同样被释放。
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.adapters.custom_provider_client import resolve_custom_provider_client_config
from app.adapters.types import AdapterInput
from app.db.models import Artifact
from app.db.session import SessionLocal
from app.errors import ServiceError
from app.schemas.events import (
    ArtifactCreateEvent,
    ArtifactRecord,
    DeployStatusEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUsageEvent,
    MessageUsageEventWrapper,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunUsageEvent,
    RunUsageEventWrapper,
    StreamEvent,
    TextAppendDelta,
    TextPart,
    ThinkingAppendDelta,
    ThinkingPart,
    ToolCallEvent,
    ToolResultEvent,
)
from app.tools.registry import tool_registry
from app.tools.types import ToolContext
from app.utils.abort import AbortSignal
from app.utils.ids import new_message_id
from app.utils.time import now_ms

logger = logging.getLogger(__name__)

# 单个 run 里 LLM 调用（tool loop 一轮算一次）的最大轮数，防死循环
MAX_TURNS = 8
# 防止单条 user message 塞太多图片（token 爆炸 + provider 通常有上限）
MAX_IMAGES_PER_MESSAGE = 5
# 网络类错误（408/429/5xx）的自动重试次数；流一旦开始吐 chunk 就不再重试
MAX_API_RETRIES = 2


class CustomAgentAdapter:
    name = "custom"

    async def stream(self, input: AdapterInput, signal: AbortSignal) -> AsyncIterator[StreamEvent]:
        if not input.customConfig:
            raise ServiceError("CustomAgentAdapter requires customConfig")
        if not input.modelId:
            raise ServiceError("CustomAgentAdapter requires modelId")

        provider = input.customConfig.get("modelProvider")
        supports_vision = bool(input.customConfig.get("supportsVision"))
        model_id = input.modelId

        client = build_client(provider, input.apiKey, input.apiBaseUrl)

        tool_defs = tool_registry.resolve(input.toolNames)
        api_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tool_defs
        ]

        ctx = ToolContext(
            conversation_id=input.conversationId,
            agent_id=input.agentId,
            run_id=input.runId,
            workspace_path=input.workspacePath,
            abort_signal=signal,
        )

        # agent 声明支持视觉且本轮真有图片 → 走 multimodal blocks，否则纯文本
        image_attachments = [
            a for a in (input.attachments or []) if a.get("kind") == "image"
        ][:MAX_IMAGES_PER_MESSAGE]
        user_content: Any = input.prompt
        if supports_vision and image_attachments:
            user_content = build_multimodal_user_content(input.prompt, image_attachments)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": input.systemPrompt},
            # 跨 run 历史（无则跳过；形状即 OpenAI ChatMessage）
            *(input.history or []),
            {"role": "user", "content": user_content},
        ]

        # 跨 turn 累加 token 用量；run 结束前 yield run.usage 给 runner 落库
        run_usage = {
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheCreationTokens": 0,
            "cacheReadTokens": 0,
            "lastInputTokens": 0,
        }

        turn = 0
        while turn < MAX_TURNS:
            if signal.aborted:
                return
            turn += 1

            message_id = new_message_id()
            yield MessageStartEvent(
                conversationId=input.conversationId,
                timestamp=now_ms(),
                messageId=message_id,
                agentId=input.agentId,
                runId=input.runId,
            )

            text_part_index = -1
            text_buffer = ""
            thinking_part_index = -1
            reasoning_buffer = ""
            next_part_index = 0
            # OpenAI 流式 tool_calls 是 index + 参数分片，必须累积完再解析
            tool_call_buffer: dict[int, dict[str, str]] = {}

            try:
                stream = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=api_tools or None,  # type: ignore[arg-type]
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except Exception:
                # 建流失败：补 message.end 再抛，让 runner 把该 message 落成终态
                yield MessageEndEvent(
                    conversationId=input.conversationId, timestamp=now_ms(), messageId=message_id
                )
                raise

            finish_reason: str | None = None
            # 本 turn 单条 message 的 usage（per-message），与 runUsage 同时维护
            msg_usage = {"inputTokens": 0, "outputTokens": 0, "cacheReadTokens": 0}

            async with stream:
                async for chunk in stream:
                    if signal.aborted:
                        return
                    # final usage chunk：choices 为空、只携 usage；
                    # DeepSeek 额外带 prompt_cache_hit_tokens / prompt_cache_miss_tokens
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        inp = _usage_field(usage, "prompt_tokens", 0)
                        out = _usage_field(usage, "completion_tokens", 0)
                        cached = _usage_field(usage, "prompt_cache_hit_tokens", None)
                        if cached is None:
                            cached = _usage_field(usage, "cached_tokens", 0)
                        msg_usage["inputTokens"] += inp
                        msg_usage["outputTokens"] += out
                        msg_usage["cacheReadTokens"] += cached
                        run_usage["inputTokens"] += inp
                        run_usage["outputTokens"] += out
                        run_usage["cacheReadTokens"] += cached
                        run_usage["lastInputTokens"] = inp

                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta

                    # reasoning_content（thinking 模型）→ thinking part 流式吐给 UI
                    reasoning_content = getattr(delta, "reasoning_content", None)
                    if isinstance(reasoning_content, str) and reasoning_content:
                        if thinking_part_index < 0:
                            thinking_part_index = next_part_index
                            next_part_index += 1
                            yield PartStartEvent(
                                conversationId=input.conversationId,
                                timestamp=now_ms(),
                                messageId=message_id,
                                partIndex=thinking_part_index,
                                part=ThinkingPart(content=""),
                            )
                        reasoning_buffer += reasoning_content
                        yield PartDeltaEvent(
                            conversationId=input.conversationId,
                            timestamp=now_ms(),
                            messageId=message_id,
                            partIndex=thinking_part_index,
                            delta=ThinkingAppendDelta(text=reasoning_content),
                        )

                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content:
                        if text_part_index < 0:
                            text_part_index = next_part_index
                            next_part_index += 1
                            yield PartStartEvent(
                                conversationId=input.conversationId,
                                timestamp=now_ms(),
                                messageId=message_id,
                                partIndex=text_part_index,
                                part=TextPart(content=""),
                            )
                        text_buffer += content
                        yield PartDeltaEvent(
                            conversationId=input.conversationId,
                            timestamp=now_ms(),
                            messageId=message_id,
                            partIndex=text_part_index,
                            delta=TextAppendDelta(text=content),
                        )

                    for tcd in getattr(delta, "tool_calls", None) or ():
                        entry = tool_call_buffer.setdefault(
                            tcd.index, {"id": "", "name": "", "args": ""}
                        )
                        if tcd.id:
                            entry["id"] = tcd.id
                        if tcd.function and tcd.function.name:
                            entry["name"] = tcd.function.name
                        if tcd.function and tcd.function.arguments:
                            entry["args"] += tcd.function.arguments

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

            if thinking_part_index >= 0:
                yield PartEndEvent(
                    conversationId=input.conversationId,
                    timestamp=now_ms(),
                    messageId=message_id,
                    partIndex=thinking_part_index,
                )
            if text_part_index >= 0:
                yield PartEndEvent(
                    conversationId=input.conversationId,
                    timestamp=now_ms(),
                    messageId=message_id,
                    partIndex=text_part_index,
                )

            tool_calls = [tc for tc in tool_call_buffer.values() if tc["id"] and tc["name"]]

            # 写回 assistant message。thinking 模型的 reasoning_content 必须一起
            # 回传下一轮，否则 API 报 "The reasoning_content ... must be passed back"
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": text_buffer or None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["args"] or "{}"},
                    }
                    for tc in tool_calls
                ]
            if reasoning_buffer:
                assistant_msg["reasoning_content"] = reasoning_buffer
            messages.append(assistant_msg)

            if not tool_calls or finish_reason == "stop":
                if msg_usage["inputTokens"] > 0 or msg_usage["outputTokens"] > 0:
                    yield MessageUsageEventWrapper(
                        conversationId=input.conversationId,
                        timestamp=now_ms(),
                        messageId=message_id,
                        usage=MessageUsageEvent(**msg_usage),
                    )
                yield MessageEndEvent(
                    conversationId=input.conversationId, timestamp=now_ms(), messageId=message_id
                )
                yield RunUsageEventWrapper(
                    conversationId=input.conversationId,
                    timestamp=now_ms(),
                    runId=input.runId,
                    usage=RunUsageEvent(model=model_id, **run_usage),
                )
                return

            # 执行工具，结果回灌
            for tc in tool_calls:
                try:
                    args = json.loads(tc["args"]) if tc["args"] else {}
                except json.JSONDecodeError:
                    args = {}

                yield ToolCallEvent(
                    conversationId=input.conversationId,
                    timestamp=now_ms(),
                    messageId=message_id,
                    callId=tc["id"],
                    toolName=tc["name"],
                    args=args,
                )

                result = await tool_registry.execute(tc["name"], args, ctx)
                value = result.value if result.ok else {"error": result.error}

                yield ToolResultEvent(
                    conversationId=input.conversationId,
                    timestamp=now_ms(),
                    messageId=message_id,
                    callId=tc["id"],
                    result=value,
                    isError=not result.ok,
                )

                # 工具只写 DB，**不发** artifact.create / deploy.status —— 由这里统一发，
                # 事件流才有单一来源；runner 收到后会往消息里注入对应的 part。
                if result.ok and isinstance(value, dict):
                    if tc["name"] == "write_artifact" and value.get("artifactId"):
                        artifact_event = await _load_artifact_event(
                            input.conversationId, value["artifactId"]
                        )
                        if artifact_event is not None:
                            yield artifact_event
                    elif tc["name"] in ("deploy_artifact", "deploy_workspace") and value.get("id"):
                        yield DeployStatusEvent(
                            conversationId=input.conversationId,
                            timestamp=now_ms(),
                            messageId=message_id,
                            deployment=value,
                        )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(value, ensure_ascii=False),
                    }
                )

            if msg_usage["inputTokens"] > 0 or msg_usage["outputTokens"] > 0:
                yield MessageUsageEventWrapper(
                    conversationId=input.conversationId,
                    timestamp=now_ms(),
                    messageId=message_id,
                    usage=MessageUsageEvent(**msg_usage),
                )
            yield MessageEndEvent(
                conversationId=input.conversationId, timestamp=now_ms(), messageId=message_id
            )
            # 继续下一轮

        # MAX_TURNS 兜底：正常路径在上面已 emit + return，这里补一次累计 usage
        yield RunUsageEventWrapper(
            conversationId=input.conversationId,
            timestamp=now_ms(),
            runId=input.runId,
            usage=RunUsageEvent(model=model_id, **run_usage),
        )


# ─── 辅助 ────────────────────────────────────────────────


async def _load_artifact_event(
    conversation_id: str, artifact_id: str
) -> ArtifactCreateEvent | None:
    """工具只返回 artifactId，事件里要带完整记录，所以回查一次 DB。"""
    async with SessionLocal() as session:
        row = await session.get(Artifact, artifact_id)
    if row is None:
        return None
    return ArtifactCreateEvent(
        conversationId=conversation_id,
        timestamp=now_ms(),
        artifact=ArtifactRecord(
            id=row.id,
            conversationId=row.conversation_id,
            type=row.type,
            title=row.title,
            content=row.content,
            version=row.version,
            parentArtifactId=row.parent_artifact_id,
            createdByAgentId=row.created_by_agent_id,
            createdAt=row.created_at,
        ),
    )


def build_client(provider: str, override_key: str | None, api_base_url: str | None) -> AsyncOpenAI:
    config = resolve_custom_provider_client_config(provider, override_key, api_base_url)
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        max_retries=MAX_API_RETRIES,
    )


def _usage_field(usage: Any, name: str, default: int | None) -> int:
    """usage 对象的取值：先看官方字段，再看 provider 私有扩展字段。"""
    value = getattr(usage, name, None)
    if value is None:
        extra = getattr(usage, "model_extra", None) or {}
        value = extra.get(name)
    if value is None:
        return default if default is not None else 0
    return int(value)


def build_multimodal_user_content(prompt: str, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI 风格多模态 content：text 块 + image_url(data URI) 块。"""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        try:
            data = Path(img["absPath"]).read_bytes()
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['mimeType']};base64,{base64.b64encode(data).decode()}"
                    },
                }
            )
        except OSError as err:
            logger.warning("failed to read image attachment %s: %s", img.get("absPath"), err)
    return blocks
