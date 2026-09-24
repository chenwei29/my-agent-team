"""fs_write 工具：在当前 workspace 内写文件（父目录自动创建）。

auto 模式直写；review 模式挂起等用户审批 —— 见 _review_write。
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, ValidationError

from app.schemas.dispatch import RunFileEvidence
from app.security.workspace_utils import assert_path_within_workspace
from app.services.dispatch_file_writes import record_file_write
from app.services.dispatch_run_evidence import record_run_file_write
from app.services.fs_service import (
    get_conversation_approval_mode,
    get_workspace_for_conversation,
    read_if_exists,
    write_file_in_workspace,
)
from app.services.pending_writes import pending_writes
from app.tools.types import ToolContext, ToolDef, ToolResult


class FsWriteArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str
    content: str


async def _handle(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = FsWriteArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid args: {err}")

    workspace = await get_workspace_for_conversation(ctx.conversation_id)
    if workspace is None:
        return ToolResult(ok=False, error="Workspace not found")

    mode = await get_conversation_approval_mode(ctx.conversation_id)

    if mode == "auto":
        return _write_now(ctx, workspace, parsed.path, parsed.content)
    return await _review_write(ctx, workspace, parsed.path, parsed.content)


def _write_now(ctx: ToolContext, workspace, path: str, content: str) -> ToolResult:
    try:
        value = write_file_in_workspace(workspace, path, content)
    except Exception as err:
        return ToolResult(ok=False, error=str(err))
    value["applied"] = "auto"
    # 落盘即记写入证据：冲突检测按 run 归档绝对路径，证据门禁按 run 索引文件改动
    record_file_write(ctx.run_id, value["absolutePath"], content)
    record_run_file_write(
        ctx.run_id,
        RunFileEvidence(
            path=path,
            absolutePath=value["absolutePath"],
            bytes=value["bytes"],
            applied="auto",
        ),
    )
    return ToolResult(ok=True, value=value)


async def _review_write(ctx: ToolContext, workspace, path: str, content: str) -> ToolResult:
    """review 模式：挂起等用户批准（审批中转在 services/pending_writes）。"""
    try:
        absolute_path = assert_path_within_workspace(workspace, path)
    except Exception as err:
        return ToolResult(ok=False, error=str(err))

    old_content = read_if_exists(workspace, path)
    pending = pending_writes.register(
        conversation_id=ctx.conversation_id,
        agent_id=ctx.agent_id,
        run_id=ctx.run_id,
        path=path,
        absolute_path=absolute_path,
        old_content=old_content,
        new_content=content,
        workspace=workspace,
    )

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    if not pending_writes.attach_resolver(pending["id"], future):
        # 用户在 attach 之前就操作了（极端竞态）：按拒绝处理
        return ToolResult(ok=False, error="User rejected the file change")

    def _on_abort() -> None:
        pending_writes.cancel(pending["id"])

    if ctx.abort_signal is not None and ctx.abort_signal.aborted:
        _on_abort()
    elif ctx.abort_signal is not None:
        ctx.abort_signal.add_listener(_on_abort)
    try:
        decision = await future
    finally:
        if ctx.abort_signal is not None:
            ctx.abort_signal.remove_listener(_on_abort)

    if not decision.get("applied"):
        return ToolResult(ok=False, error="User rejected the file change")

    record_file_write(ctx.run_id, absolute_path, content)
    record_run_file_write(
        ctx.run_id,
        RunFileEvidence(
            path=path,
            absolutePath=absolute_path,
            bytes=len(content.encode("utf-8")),
            applied="review",
        ),
    )

    return ToolResult(
        ok=True,
        value={
            "path": path,
            "absolutePath": absolute_path,
            "bytes": len(content.encode("utf-8")),
            "applied": "review",
        },
    )


FS_WRITE_TOOL = ToolDef(
    name="fs_write",
    description=(
        "Write (create or overwrite) a file inside the current workspace. "
        "Parent directories are created automatically. Max 100KB per write."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace root"},
            "content": {"type": "string", "description": "Full file content to write"},
        },
        "required": ["path", "content"],
    },
    handler=_handle,
)
