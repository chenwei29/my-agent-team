"""CustomAgentAdapter：chunk → StreamEvent 映射、tool loop、usage、中止。

不联网：monkeypatch build_client 换成假客户端，喂手工构造的流式 chunk。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.adapters import custom as custom_module
from app.adapters.custom import CustomAgentAdapter
from app.adapters.types import AdapterInput
from app.errors import ServiceError
from app.tools.types import ToolDef, ToolResult
from app.utils.abort import AbortSignal


# ─── 假 OpenAI 客户端 ──────────────────────────────────────


class FakeStream:
    def __init__(self, chunks: list[Any], on_chunk=None) -> None:
        self._chunks = iter(chunks)
        self._on_chunk = on_chunk
        self.closed = False

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self):
        if self._on_chunk is not None:
            self._on_chunk()
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self) -> None:
        self.closed = True


class FakeCompletions:
    def __init__(self, responses: list[list[Any]]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(next(self._responses))


class FakeClient:
    def __init__(self, responses: list[list[Any]]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def _chunk(content: str | None = None, reasoning: str | None = None,
           tool_calls: list | None = None, finish_reason: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _usage_chunk(prompt_tokens: int, completion_tokens: int, **extra: int) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, **extra)
    return SimpleNamespace(choices=[], usage=usage)


def _tool_delta(index: int, call_id: str | None, name: str | None, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _input(**overrides: Any) -> AdapterInput:
    base = dict(
        agentId="ag_test",
        conversationId="conv_test",
        runId="run_test",
        prompt="你好",
        workspacePath="/tmp/ws",
        systemPrompt="你是测试 agent",
        apiKey="test-key",
        apiBaseUrl="https://example.com/v1",
        modelId="test-model",
        toolNames=[],
        customConfig={"modelProvider": "openai-compatible", "supportsVision": False},
    )
    base.update(overrides)
    return AdapterInput(**base)


async def _collect(adapter: CustomAgentAdapter, inp: AdapterInput, signal: AbortSignal):
    return [event async for event in adapter.stream(inp, signal)]


def _install(monkeypatch: pytest.MonkeyPatch, responses: list[list[Any]]) -> FakeClient:
    client = FakeClient(responses)
    monkeypatch.setattr(custom_module, "build_client", lambda *a: client)
    return client


# ─── 纯文本流式 ──────────────────────────────────────────


async def test_text_streaming_and_usage(monkeypatch: pytest.MonkeyPatch):
    client = _install(
        monkeypatch,
        [
            [
                _chunk(content="你"),
                _chunk(content="好"),
                _chunk(finish_reason="stop"),
                _usage_chunk(12, 34, prompt_cache_hit_tokens=5),
            ]
        ],
    )

    events = await _collect(CustomAgentAdapter(), _input(), AbortSignal())

    types = [e.type for e in events]
    assert types == [
        "message.start",
        "part.start",
        "part.delta",
        "part.delta",
        "part.end",
        "message.usage",
        "message.end",
        "run.usage",
    ]

    part_start = events[1]
    assert part_start.partIndex == 0
    assert part_start.part.type == "text"
    assert events[2].delta.type == "text.append"
    assert events[2].delta.text == "你"

    message_usage = events[5].usage
    assert message_usage.inputTokens == 12
    assert message_usage.outputTokens == 34
    assert message_usage.cacheReadTokens == 5  # DeepSeek prompt_cache_hit_tokens 映射

    run_usage = events[7].usage
    assert run_usage.inputTokens == 12
    assert run_usage.outputTokens == 34
    assert run_usage.cacheReadTokens == 5
    assert run_usage.lastInputTokens == 12
    assert run_usage.model == "test-model"

    # 请求形状：system 在最前，user 在其后（run 结束后 messages 会追加 assistant，别用 -1）
    messages = client.chat.completions.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "你是测试 agent"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "你好"
    # 没有注册工具时不带 tools
    assert not client.chat.completions.calls[0]["tools"]


async def test_history_injected_between_system_and_user(monkeypatch: pytest.MonkeyPatch):
    client = _install(monkeypatch, [[_chunk(content="好", finish_reason="stop")]])

    inp = _input(history=[{"role": "user", "content": "第一轮"}, {"role": "assistant", "content": "收到"}])
    await _collect(CustomAgentAdapter(), inp, AbortSignal())

    messages = client.chat.completions.calls[0]["messages"]
    # run 结束后 messages 末尾追加了 assistant；看初始请求形状取前 3 条
    assert [m["role"] for m in messages[:3]] == ["system", "user", "assistant"]
    assert messages[1]["content"] == "第一轮"
    assert messages[3]["role"] == "user"  # 当前触发消息


async def test_reasoning_content_maps_to_thinking_part(monkeypatch: pytest.MonkeyPatch):
    _install(
        monkeypatch,
        [[
            _chunk(reasoning="先想一想"),
            _chunk(content="答案是 42"),
            _chunk(finish_reason="stop"),
        ]],
    )

    events = await _collect(CustomAgentAdapter(), _input(), AbortSignal())

    thinking_starts = [e for e in events if e.type == "part.start" and e.part.type == "thinking"]
    text_starts = [e for e in events if e.type == "part.start" and e.part.type == "text"]
    assert len(thinking_starts) == 1
    assert len(text_starts) == 1
    assert thinking_starts[0].partIndex == 0
    assert text_starts[0].partIndex == 1

    thinking_deltas = [e for e in events if e.type == "part.delta" and e.delta.type == "thinking.append"]
    assert thinking_deltas[0].delta.text == "先想一想"


# ─── tool loop ────────────────────────────────────────────


async def test_tool_call_accumulation_and_loop(monkeypatch: pytest.MonkeyPatch):
    from app.tools.registry import tool_registry

    seen_args: list[dict] = []

    async def echo_handler(args: dict, ctx) -> ToolResult:
        seen_args.append({"args": args, "ctx": ctx})
        return ToolResult(ok=True, value={"echo": args.get("value")})

    tool_registry.register(
        ToolDef(
            name="echo_tool",
            description="回声测试工具",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}},
            handler=echo_handler,
        )
    )
    try:
        client = _install(
            monkeypatch,
            [
                # 第一轮：参数分片跨多个 chunk 到达
                [
                    _chunk(tool_calls=[_tool_delta(0, "call_1", "echo_tool", '{"value": ')]),
                    _chunk(tool_calls=[_tool_delta(0, None, None, '"hi"}')]),
                    _chunk(finish_reason="tool_calls"),
                    _usage_chunk(10, 5),
                ],
                # 第二轮：模型收到工具结果后给出最终答复
                [_chunk(content="完成", finish_reason="stop"), _usage_chunk(20, 7)],
            ],
        )

        inp = _input(toolNames=["echo_tool"])
        events = await _collect(CustomAgentAdapter(), inp, AbortSignal())
        types = [e.type for e in events]

        # 工具调用 → 结果 → 第二轮 message → run.usage
        tool_call = next(e for e in events if e.type == "tool.call")
        assert tool_call.callId == "call_1"
        assert tool_call.toolName == "echo_tool"
        assert tool_call.args == {"value": "hi"}  # 分片累积后才 parse

        tool_result = next(e for e in events if e.type == "tool.result")
        assert tool_result.result == {"echo": "hi"}
        assert tool_result.isError is False

        # ctx 从 AdapterInput 透传
        assert seen_args[0]["ctx"].conversation_id == "conv_test"
        assert seen_args[0]["ctx"].run_id == "run_test"

        # 第二轮请求把 assistant(tool_calls) + tool 结果回灌：
        # [system, user, assistant(tool_calls), tool]，结束后又追加最终 assistant
        second_messages = client.chat.completions.calls[1]["messages"]
        assert [m["role"] for m in second_messages] == [
            "system", "user", "assistant", "tool", "assistant",
        ]
        assistant = second_messages[2]
        assert assistant["tool_calls"][0]["function"]["name"] == "echo_tool"
        assert assistant["tool_calls"][0]["function"]["arguments"] == '{"value": "hi"}'
        tool_msg = second_messages[3]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"
        assert json.loads(tool_msg["content"]) == {"echo": "hi"}

        # run.usage 是两轮累加
        run_usage = next(e for e in events if e.type == "run.usage").usage
        assert run_usage.inputTokens == 30
        assert run_usage.outputTokens == 12

        # 两轮各自 message.start / message.end
        assert types.count("message.start") == 2
        assert types.count("message.end") == 2
    finally:
        tool_registry._tools.pop("echo_tool", None)


async def test_tool_error_becomes_error_result(monkeypatch: pytest.MonkeyPatch):
    from app.tools.registry import tool_registry

    async def bad_handler(args: dict, ctx) -> ToolResult:
        raise RuntimeError("工具炸了")

    tool_registry.register(
        ToolDef(name="bad_tool", description="", parameters={}, handler=bad_handler)
    )
    try:
        _install(
            monkeypatch,
            [
                [_chunk(tool_calls=[_tool_delta(0, "call_9", "bad_tool", "")], finish_reason="tool_calls")],
                [_chunk(content="算了", finish_reason="stop")],
            ],
        )
        inp = _input(toolNames=["bad_tool"])
        events = await _collect(CustomAgentAdapter(), inp, AbortSignal())

        tool_result = next(e for e in events if e.type == "tool.result")
        assert tool_result.isError is True
        assert tool_result.result == {"error": "工具炸了"}
        # 空参数串 parse 成 {}
        assert next(e for e in events if e.type == "tool.call").args == {}
    finally:
        tool_registry._tools.pop("bad_tool", None)


# ─── 中止 ────────────────────────────────────────────────


async def test_abort_before_stream_returns_nothing(monkeypatch: pytest.MonkeyPatch):
    client = _install(monkeypatch, [[_chunk(content="x")]])
    signal = AbortSignal()
    signal.abort()

    events = await _collect(CustomAgentAdapter(), _input(), signal)
    assert events == []
    assert client.chat.completions.calls == []  # 建流都没发生


async def test_abort_mid_stream_closes_stream(monkeypatch: pytest.MonkeyPatch):
    streams: list[FakeStream] = []
    signal = AbortSignal()
    state = {"count": 0}

    class _TrackedClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            chunks = [_chunk(content="a"), _chunk(content="b"), _chunk(finish_reason="stop")]

            def on_chunk():
                state["count"] += 1
                if state["count"] >= 2:  # 第二个 chunk 到达前后触发 abort
                    signal.abort()

            stream = FakeStream(chunks, on_chunk=on_chunk)
            streams.append(stream)
            return stream

    monkeypatch.setattr(custom_module, "build_client", lambda *args: _TrackedClient())

    events = await _collect(CustomAgentAdapter(), _input(), signal)

    # 已产出的 part 保留，但不再有 part.end / message.end / run.usage
    types = [e.type for e in events]
    assert types[0] == "message.start"
    assert "part.delta" in types
    assert "part.end" not in types
    assert "message.end" not in types
    assert "run.usage" not in types
    # HTTP 流被关闭（连接归还）
    assert streams[0].closed is True


# ─── 前置校验 ─────────────────────────────────────────────


async def test_requires_custom_config_and_model_id():
    adapter = CustomAgentAdapter()
    with pytest.raises(ServiceError, match="requires customConfig"):
        async for _ in adapter.stream(_input(customConfig=None), AbortSignal()):
            pass
    with pytest.raises(ServiceError, match="requires modelId"):
        async for _ in adapter.stream(_input(modelId=None), AbortSignal()):
            pass


# ─── 多模态 ──────────────────────────────────────────────


async def test_multimodal_when_supports_vision(monkeypatch: pytest.MonkeyPatch, tmp_path):
    client = _install(monkeypatch, [[_chunk(content="看到了", finish_reason="stop")]])

    image = tmp_path / "cat.png"
    image.write_bytes(b"\x89PNG fake")

    inp = _input(
        customConfig={"modelProvider": "openai-compatible", "supportsVision": True},
        attachments=[
            {"kind": "image", "absPath": str(image), "mimeType": "image/png", "id": "att_1", "fileName": "cat.png"},
            {"kind": "file", "absPath": str(tmp_path / "a.txt"), "mimeType": "text/plain", "id": "att_2", "fileName": "a.txt"},
        ],
    )
    await _collect(CustomAgentAdapter(), inp, AbortSignal())

    content = client.chat.completions.calls[0]["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "你好"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_no_multimodal_without_vision(monkeypatch: pytest.MonkeyPatch, tmp_path):
    client = _install(monkeypatch, [[_chunk(content="看不到", finish_reason="stop")]])

    inp = _input(
        attachments=[{"kind": "image", "absPath": str(tmp_path / "x.png"), "mimeType": "image/png"}],
    )
    await _collect(CustomAgentAdapter(), inp, AbortSignal())

    content = client.chat.completions.calls[0]["messages"][1]["content"]
    assert content == "你好"
