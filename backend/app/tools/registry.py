"""工具注册表：register / resolve / execute。

resolve 按 agent 配置的 toolNames 过滤；未注册的名字直接跳过（agent 配置里
留着已下线工具名时 run 不应该崩）。execute 捕获一切异常并转成失败 ToolResult。
"""

from __future__ import annotations

import logging

from app.tools.types import ToolContext, ToolDef, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def resolve(self, names: list[str]) -> list[ToolDef]:
        return [self._tools[n] for n in names if n in self._tools]

    async def execute(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None or tool.handler is None:
            return ToolResult(ok=False, error=f"Unknown tool: {name}")
        try:
            return await tool.handler(args, ctx)
        except Exception as err:  # noqa: BLE001 - 工具异常不允许炸掉 run
            logger.exception("tool %s failed", name)
            return ToolResult(ok=False, error=str(err))


# 模块级单例：工具都在本进程内注册，无跨进程需求
tool_registry = ToolRegistry()
