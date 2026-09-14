"""MockAdapter —— 不调用真实 LLM 的假实现，事件序列与真实 adapter 同构。

按 prompt 关键词选脚本，字符级流式吐事件（每 15–20ms 一个 chunk）。
用途是端到端骨架验证（SSE / store / 打字机渲染）与 e2e 的确定性回复。

行为要点（改这里前先对齐下面这些约束）：
- 关键词顺序：greeting → code → tool → default，**第一个匹配即返回**；
- chunk 大小：text 4、thinking 8、code 8；sleep：text 20ms、thinking/code 15ms、tool 300ms；
- sleep 在**每个** chunk 之前（含第一个）；
- `partIndex` 是**脚本步序号**（含 tool 步），不是 parts 数组下标；
- tool 步只发 `tool.call`/`tool.result`，不发任何 part 事件；
- abort 只在循环头检查（每个 step / 每个 chunk），已进入的 sleep 会睡完；
- abort 之后仍然**无条件**发 `part.end` 与 `message.end`；
- 不发 `run.usage` / `message.usage`，所以 run/message 的 usage 保持 NULL。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.adapters.types import AdapterInput
from app.schemas.events import (
    CodeAppendDelta,
    CodePart,
    MessageEndEvent,
    MessageStartEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    StreamEvent,
    TextAppendDelta,
    TextPart,
    ThinkingAppendDelta,
    ThinkingPart,
    ToolCallEvent,
    ToolResultEvent,
)
from app.utils.abort import AbortSignal
from app.utils.ids import new_message_id, new_tool_call_id
from app.utils.time import now_ms

# ─── 脚本步骤 ───────────────────────────────────────────────


@dataclass
class TextStep:
    content: str


@dataclass
class ThinkingStep:
    content: str


@dataclass
class CodeStep:
    language: str
    content: str


@dataclass
class ToolStep:
    toolName: str
    args: Any
    result: Any = None


ScriptStep = TextStep | ThinkingStep | CodeStep | ToolStep

# ─── 内置脚本（文案是 e2e 断言依赖的固定字符串，改动会挂测试）───

GREETING_SCRIPT: list[ScriptStep] = [
    ThinkingStep("用户在问候，我应该礼貌回应并介绍自己。"),
    TextStep(
        "你好！我是 Mock Agent，目前用于验证 AgentHub 的端到端骨架。"
        "我会按预设脚本流式回复，不消耗任何 LLM token。\n\n"
        "你可以试试输入「写代码」或「执行任务」看其他场景。"
    ),
]

CODE_SCRIPT: list[ScriptStep] = [
    ThinkingStep("用户希望看到代码示例，我演示一段 React 组件代码。"),
    TextStep("好的，这是一个简单的 React 计数器组件："),
    CodeStep(
        "tsx",
        """import { useState } from 'react'

export function Counter() {
  const [count, setCount] = useState(0)
  return (
    <div className="flex items-center gap-2">
      <button onClick={() => setCount(c => c - 1)}>-</button>
      <span>{count}</span>
      <button onClick={() => setCount(c => c + 1)}>+</button>
    </div>
  )
}""",
    ),
    TextStep("这是最朴素的实现。需要扩展可以告诉我（持久化、键盘快捷键等）。"),
]

TOOL_SCRIPT: list[ScriptStep] = [
    ThinkingStep("我需要演示工具调用流程。"),
    TextStep("我先调用工具收集信息："),
    ToolStep("read_artifact", {"artifactId": "art_demo"}, {"title": "示例产物", "size": 1024}),
    TextStep("已读取产物信息。这只是脚本演示，真实工具会在后续 milestone 接入。"),
]

DEFAULT_SCRIPT: list[ScriptStep] = [
    ThinkingStep("收到了用户消息，按通用模板回应。"),
    TextStep(
        "我收到了你的消息。这是 MockAdapter 的默认响应，"
        "用于验证消息流式渲染、part 切换、tool 调用等链路。\n\n"
        '试试输入 "你好"、"写代码"、"执行任务" 触发不同脚本。'
    ),
]

# 正则不加词边界，第一个匹配即返回 —— 别"修"成更精确的匹配：
# "thinking about this" 含 hi → greeting；"decode" 含 code → code；"truncate" 含 run → tool。
_GREETING_RE = re.compile(r"(你好|hello|hi|您好)")
_CODE_RE = re.compile(r"(写代码|代码|code|component|组件)")
_TOOL_RE = re.compile(r"(执行|工具|tool|run|跑)")


def pick_script(prompt: str) -> list[ScriptStep]:
    p = prompt.lower()
    if _GREETING_RE.search(p):
        return GREETING_SCRIPT
    if _CODE_RE.search(p):
        return CODE_SCRIPT
    if _TOOL_RE.search(p):
        return TOOL_SCRIPT
    return DEFAULT_SCRIPT


def _chunk_text(text: str, size: int) -> list[str]:
    """按 Unicode 码点切片 —— BMP 文本与按 UTF-16 code unit 切片等价，
    仅当文案里出现 emoji 等星平面字符时 chunk 边界会不同（内置脚本没有）。"""
    return [text[i : i + size] for i in range(0, len(text), size)]


async def _sleep(ms: int) -> None:
    await asyncio.sleep(ms / 1000)


class MockAdapter:
    name = "mock"

    async def stream(self, input: AdapterInput, signal: AbortSignal) -> AsyncIterator[StreamEvent]:
        script = pick_script(input.prompt)

        message_id = new_message_id()
        conv = input.conversationId
        yield MessageStartEvent(
            conversationId=conv,
            timestamp=now_ms(),
            messageId=message_id,
            agentId=input.agentId,
            runId=input.runId,
        )

        part_index = -1

        for step in script:
            if signal.aborted:
                break

            part_index += 1

            if isinstance(step, TextStep):
                yield PartStartEvent(
                    conversationId=conv,
                    timestamp=now_ms(),
                    messageId=message_id,
                    partIndex=part_index,
                    part=TextPart(content=""),
                )
                for chunk in _chunk_text(step.content, 4):
                    if signal.aborted:
                        break
                    await _sleep(20)
                    yield PartDeltaEvent(
                        conversationId=conv,
                        timestamp=now_ms(),
                        messageId=message_id,
                        partIndex=part_index,
                        delta=TextAppendDelta(text=chunk),
                    )
                yield PartEndEvent(
                    conversationId=conv,
                    timestamp=now_ms(),
                    messageId=message_id,
                    partIndex=part_index,
                )

            elif isinstance(step, ThinkingStep):
                yield PartStartEvent(
                    conversationId=conv,
                    timestamp=now_ms(),
                    messageId=message_id,
                    partIndex=part_index,
                    part=ThinkingPart(content=""),
                )
                for chunk in _chunk_text(step.content, 8):
                    if signal.aborted:
                        break
                    await _sleep(15)
                    yield PartDeltaEvent(
                        conversationId=conv,
                        timestamp=now_ms(),
                        messageId=message_id,
                        partIndex=part_index,
                        delta=ThinkingAppendDelta(text=chunk),
                    )
                yield PartEndEvent(
                    conversationId=conv,
                    timestamp=now_ms(),
                    messageId=message_id,
                    partIndex=part_index,
                )

            elif isinstance(step, CodeStep):
                yield PartStartEvent(
                    conversationId=conv,
                    timestamp=now_ms(),
                    messageId=message_id,
                    partIndex=part_index,
                    part=CodePart(language=step.language, content=""),
                )
                for chunk in _chunk_text(step.content, 8):
                    if signal.aborted:
                        break
                    await _sleep(15)
                    yield PartDeltaEvent(
                        conversationId=conv,
                        timestamp=now_ms(),
                        messageId=message_id,
                        partIndex=part_index,
                        delta=CodeAppendDelta(text=chunk),
                    )
                yield PartEndEvent(
                    conversationId=conv,
                    timestamp=now_ms(),
                    messageId=message_id,
                    partIndex=part_index,
                )

            elif isinstance(step, ToolStep):
                call_id = new_tool_call_id()
                yield ToolCallEvent(
                    conversationId=conv,
                    timestamp=now_ms(),
                    messageId=message_id,
                    callId=call_id,
                    toolName=step.toolName,
                    args=step.args,
                )
                await _sleep(300)
                yield ToolResultEvent(
                    conversationId=conv,
                    timestamp=now_ms(),
                    messageId=message_id,
                    callId=call_id,
                    result=step.result if step.result is not None else {"ok": True},
                    isError=False,
                )

        yield MessageEndEvent(conversationId=conv, timestamp=now_ms(), messageId=message_id)
