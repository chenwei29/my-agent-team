"""fs_list 工具：列当前 workspace 目录。dotfile 隐藏、目录优先。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationError

from app.services.fs_service import get_workspace_for_conversation, list_dir_in_workspace
from app.tools.types import ToolContext, ToolDef, ToolResult


class FsListArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = ""


async def _handle(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = FsListArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid args: {err}")

    workspace = await get_workspace_for_conversation(ctx.conversation_id)
    if workspace is None:
        return ToolResult(ok=False, error="Workspace not found")

    try:
        value = list_dir_in_workspace(workspace, parsed.path)
    except Exception as err:
        return ToolResult(ok=False, error=str(err))
    return ToolResult(ok=True, value=value)


FS_LIST_TOOL = ToolDef(
    name="fs_list",
    description="List entries of a directory inside the current workspace. Omit path to list the workspace root.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path relative to the workspace root (optional)"},
        },
    },
    handler=_handle,
)
