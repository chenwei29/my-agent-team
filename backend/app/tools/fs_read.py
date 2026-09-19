"""fs_read 工具：在当前 workspace 内读文件。

大小 / 字符数有硬限制（1MB / 50000 字符），超出报错或截断 —— 限制本身是
给 LLM 的行为契约（让它学会分块读大文件），错误文案会原样回灌给 LLM。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationError

from app.services.fs_service import get_workspace_for_conversation, read_file_in_workspace
from app.tools.types import ToolContext, ToolDef, ToolResult


class FsReadArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str


async def _handle(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = FsReadArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid args: {err}")

    workspace = await get_workspace_for_conversation(ctx.conversation_id)
    if workspace is None:
        return ToolResult(ok=False, error="Workspace not found")

    try:
        value = read_file_in_workspace(workspace, parsed.path)
    except Exception as err:  # 限额 / 非文件 / 逃逸的文案都是契约，原样给 LLM
        return ToolResult(ok=False, error=str(err))
    return ToolResult(ok=True, value=value)


FS_READ_TOOL = ToolDef(
    name="fs_read",
    description=(
        "Read a file from the current workspace. "
        "Content longer than 50,000 characters is truncated (the result is marked truncated=true). "
        "Files larger than 1MB cannot be read and must be read in chunks if possible."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace root"},
        },
        "required": ["path"],
    },
    handler=_handle,
)
