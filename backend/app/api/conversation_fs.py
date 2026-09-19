"""会话内文件端点（FileTab 用）：listdir / read / write。

用户手动编辑保存**不走审批**（审批只约束 agent 的 fs_write 工具）——
这是有意的契约：人在界面上改文件本身就是一次显式操作。

错误状态码按错误消息子串映射：outside→403、too large/quota→413、
Not a→400、其余 500。这套派生规则与错误文案耦合，文案改动要同步这里。
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.errors import HttpError
from app.schemas.entities import (
    WorkspaceFsWriteBody,
    WorkspaceListResult,
    WorkspaceReadResult,
    WorkspaceWriteResult,
)
from app.security.workspace_utils import PathOutsideWorkspaceError
from app.services.fs_service import (
    get_workspace_for_conversation,
    list_dir_in_workspace,
    read_file_in_workspace,
    write_file_in_workspace,
)

router = APIRouter(prefix="/api/conversations/{conversation_id}/fs", tags=["conversation-fs"])


def _dispatch_error(err: Exception) -> None:
    message = str(err)
    if isinstance(err, PathOutsideWorkspaceError) or "outside" in message:
        raise HttpError(403, message) from err
    if "too large" in message or "quota" in message:
        raise HttpError(413, message) from err
    if "Not a" in message:
        raise HttpError(400, message) from err
    raise HttpError(500, message) from err


@router.get("/listdir", response_model=WorkspaceListResult)
async def list_dir(conversation_id: str, path: str = Query(default="")) -> dict:
    workspace = await get_workspace_for_conversation(conversation_id)
    if workspace is None:
        raise HttpError(404, "Workspace not found")
    try:
        return list_dir_in_workspace(workspace, path)
    except Exception as err:
        _dispatch_error(err)


@router.get("/read", response_model=WorkspaceReadResult)
async def read_file(conversation_id: str, path: str | None = Query(default=None)) -> dict:
    if not path:
        raise HttpError(400, "path required")
    workspace = await get_workspace_for_conversation(conversation_id)
    if workspace is None:
        raise HttpError(404, "Workspace not found")
    try:
        return read_file_in_workspace(workspace, path)
    except Exception as err:
        _dispatch_error(err)


@router.post("/write", response_model=WorkspaceWriteResult)
async def write_file(conversation_id: str, body: WorkspaceFsWriteBody) -> dict:
    workspace = await get_workspace_for_conversation(conversation_id)
    if workspace is None:
        raise HttpError(404, "Workspace not found")
    try:
        return write_file_in_workspace(workspace, body.path, body.content)
    except Exception as err:
        _dispatch_error(err)
