"""MockAdapter 行为测试。

重点验证「事件序列 + 节奏 + 中止语义」，因为前端的打字机效果完全由 delta 粒度决定。
"""

from __future__ import annotations

import time

from app.adapters.mock import (
    CODE_SCRIPT,
    DEFAULT_SCRIPT,
    GREETING_SCRIPT,
    TOOL_SCRIPT,
    MockAdapter,
    pick_script,
)
from app.adapters.types import AdapterInput
from app.schemas.events import (
    CodePart,
    MessageEndEvent,
    MessageStartEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallEvent,
    ToolResultEvent,
)
from app.utils.abort import AbortSignal


def _input(prompt: str) -> AdapterInput:
    return AdapterInput(
        agentId="ag_1",
        conversationId="conv_1",
        runId="run_1",
        prompt=prompt,
        workspacePath="/tmp/ws",
        systemPrompt="system",
    )


async def _collect(prompt: str, signal: AbortSignal | None = None) -> list:
    adapter = MockAdapter()
    return [event async for event in adapter.stream(_input(prompt), signal or AbortSignal())]


def _joined_text(events: list) -> str:
    return "".join(
        e.delta.text
        for e in events
        if isinstance(e, PartDeltaEvent) and e.delta.type == "text.append"
    )


async def test_greeting_script_streams_thinking_then_text():
    events = await _collect("你好")

    assert isinstance(events[0], MessageStartEvent)
    assert events[0].agentId == "ag_1"
    assert events[0].runId == "run_1"
    assert events[0].messageId.startswith("msg_")

    kinds = [e.type for e in events]
    assert kinds[0] == "message.start"
    assert kinds[-1] == "message.end"
    assert "part.start" in kinds and "part.delta" in kinds and "part.end" in kinds

    starts = [e for e in events if isinstance(e, PartStartEvent)]
    assert [e.partIndex for e in starts] == [0, 1]
    assert isinstance(starts[0].part, ThinkingPart)
    assert isinstance(starts[1].part, TextPart)

    assert "我是 Mock Agent" in _joined_text(events)


async def test_code_script_emits_a_tsx_code_part():
    events = await _collect("写代码")

    starts = [e for e in events if isinstance(e, PartStartEvent)]
    parts = [e.part for e in starts]
    code_parts = [p for p in parts if isinstance(p, CodePart)]
    assert len(code_parts) == 1
    assert code_parts[0].language == "tsx"

    code_text = "".join(
        e.delta.text
        for e in events
        if isinstance(e, PartDeltaEvent) and e.delta.type == "code.append"
    )
    assert "React 计数器" in _joined_text(events)
    assert "export function Counter()" in code_text


async def test_text_chunks_are_four_chars_and_thinking_eight():
    events = await _collect("你好")
    text_deltas = [
        e.delta.text for e in events if isinstance(e, PartDeltaEvent) and e.delta.type == "text.append"
    ]
    thinking_deltas = [
        e.delta.text
        for e in events
        if isinstance(e, PartDeltaEvent) and e.delta.type == "thinking.append"
    ]

    assert all(len(chunk) <= 4 for chunk in text_deltas)
    assert text_deltas[0] == GREETING_SCRIPT[1].content[:4]
    assert all(len(chunk) <= 8 for chunk in thinking_deltas)
    # 前 N-1 个必须是满 chunk（最后一个才是余数）
    assert all(len(chunk) == 8 for chunk in thinking_deltas[:-1])


async def test_part_end_follows_every_part():
    events = await _collect("你好")
    started = [
        (e.messageId, e.partIndex) for e in events if isinstance(e, PartStartEvent)
    ]
    ended = [(e.messageId, e.partIndex) for e in events if isinstance(e, PartEndEvent)]
    assert started == ended


async def test_tool_step_emits_call_and_result_without_part_events():
    events = await _collect("执行任务")
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]

    assert len(calls) == 1 and len(results) == 1
    assert calls[0].toolName == "read_artifact"
    assert calls[0].callId.startswith("call_")
    assert calls[0].callId == results[0].callId
    assert results[0].isError is False
    # tool 步不发 part 事件，但**占用一个 partIndex**（按脚本步计数）：
    # 脚本是 thinking(0) → text(1) → tool(2) → text(3)，所以 part 下标是 [0, 1, 3]
    starts = [e.partIndex for e in events if isinstance(e, PartStartEvent)]
    assert starts == [0, 1, 3]


async def test_no_usage_events_from_mock():
    events = await _collect("你好")
    assert "run.usage" not in {e.type for e in events}
    assert "message.usage" not in {e.type for e in events}


def test_pick_script_keyword_order_and_substring_quirks():
    # 第一个匹配即返回：同时命中 greeting 和 code 时走 greeting
    assert pick_script("你好，写代码") is GREETING_SCRIPT
    # 关键词匹配没有词边界，这些"误命中"是可观察行为，别修
    assert pick_script("thinking about this") is GREETING_SCRIPT  # 含 "hi"
    assert pick_script("decode") is CODE_SCRIPT  # 含 "code"
    assert pick_script("truncate") is TOOL_SCRIPT  # 含 "run"
    assert pick_script("随便聊聊") is DEFAULT_SCRIPT


async def test_abort_still_closes_parts_and_ends_message():
    """中止时照样 yield part.end 与 message.end。"""
    signal = AbortSignal()
    events: list = []
    adapter = MockAdapter()
    async for event in adapter.stream(_input("你好"), signal):
        events.append(event)
        if len(events) == 4:
            signal.abort()

    assert isinstance(events[-1], MessageEndEvent)
    assert any(isinstance(e, PartEndEvent) for e in events)
    # 被中止 → 文本没吐完
    assert "我是 Mock Agent" not in _joined_text(events)


async def test_typing_pace_is_real_not_instant():
    """每次 delta 前有 sleep（text 20ms / thinking 15ms），一次发完就没有打字机效果。"""
    started = time.monotonic()
    await _collect("你好")
    assert time.monotonic() - started >= 0.3
