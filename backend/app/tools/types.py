"""工具的类型定义：ToolDef / ToolContext / ToolResult。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """工具执行结果。ok=False 时 error 会被回灌给 LLM（以及展示在前端工具卡上）。"""

    ok: bool
    value: Any = None
    error: str | None = None


@dataclass
class ToolContext:
    """一次 run 内工具执行所需的上下文。P4 引入沙箱 / 审批后字段会补齐。"""

    conversation_id: str
    agent_id: str
    run_id: str
    workspace_path: str
    # 审批中转（fs_write review 模式等）需要挂起等待用户操作，P4 接入
    abort_signal: Any = None


@dataclass
class ToolDef:
    """一个工具的自描述：name + 给 LLM 看的 description + JSON Schema parameters。"""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Any = None  # async (args: dict, ctx: ToolContext) -> ToolResult
