"""内置工具集中注册。

`register_builtin_tools()` 幂等（重复调用无副作用），由 main.py 模块级调用一次。
工具按会话粒度由 agent.tool_names 圈选；这里只负责把全部内置工具放进注册表。
"""

from __future__ import annotations

from app.tools.artifacts import READ_ARTIFACT_TOOL, WRITE_ARTIFACT_TOOL
from app.tools.ask_user import ASK_USER_TOOL
from app.tools.bash import BASH_TOOL
from app.tools.fs_list import FS_LIST_TOOL
from app.tools.fs_read import FS_READ_TOOL
from app.tools.fs_write import FS_WRITE_TOOL
from app.tools.plan_tasks import PLAN_TASKS_TOOL
from app.tools.read_attachment import READ_ATTACHMENT_TOOL
from app.tools.registry import tool_registry

_BUILTIN_TOOLS = (
    FS_READ_TOOL,
    FS_LIST_TOOL,
    FS_WRITE_TOOL,
    BASH_TOOL,
    ASK_USER_TOOL,
    READ_ATTACHMENT_TOOL,
    READ_ARTIFACT_TOOL,
    WRITE_ARTIFACT_TOOL,
    PLAN_TASKS_TOOL,
)

_registered = False


def register_builtin_tools() -> None:
    global _registered
    if _registered:
        return
    for tool in _BUILTIN_TOOLS:
        tool_registry.register(tool)
    _registered = True
