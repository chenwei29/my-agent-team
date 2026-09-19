"""Agent 可调用的工具：接口定义与注册表骨架。

工具 = LLM 通过 tool_calls 触发、后端在本进程内执行的函数。P3 只立骨架
（adapter 的 tool loop 依赖 resolve/execute 两个入口），实体工具（fs / bash /
artifact / ask_user …）后续阶段逐个注册进来。

设计约束（改这里前先想清楚）：
- ToolResult 是**唯一**的返回形态：handler 抛异常也由 registry 捕获转成
  `ToolResult(ok=False)`，绝不让工具异常炸掉整个 run；
- 工具执行永远发生在 runner 的进程内，上下文（哪个会话 / 哪个 workspace）由
  ToolContext 携带，工具自己不去翻全局状态。
"""

from app.tools.registry import ToolRegistry, tool_registry
from app.tools.types import ToolContext, ToolDef, ToolResult

__all__ = ["ToolContext", "ToolDef", "ToolResult", "ToolRegistry", "tool_registry"]
