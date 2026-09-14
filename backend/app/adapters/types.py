"""Adapter 契约：所有 agent 适配器实现的接口。

只有一个方法：`stream(input, signal) -> AsyncIterator[StreamEvent]`。
- 生成器式异步迭代器（yield 事件，正常 return 表示结束，抛异常表示硬失败）；
- 中止靠第二个位置参数 `signal` 自行轮询，没有 abort reason / cancellation token；
- adapter **永远不写 DB**（分层铁律），落库是 runner 的事。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.schemas.events import StreamEvent
from app.utils.abort import AbortSignal


@dataclass
class AdapterInput:
    """一次 run 传给 adapter 的输入（P2 只有 mock 会读 prompt / conversationId / agentId / runId）。"""

    agentId: str
    conversationId: str
    runId: str
    prompt: str
    workspacePath: str
    systemPrompt: str
    apiKey: str | None = None
    apiBaseUrl: str | None = None
    modelId: str | None = None
    toolNames: list[str] = field(default_factory=list)
    attachments: list[dict[str, Any]] | None = None
    history: list[dict[str, Any]] | None = None
    customConfig: dict[str, Any] | None = None


class AgentPlatformAdapter(Protocol):
    name: str

    def stream(self, input: AdapterInput, signal: AbortSignal) -> AsyncIterator[StreamEvent]: ...
