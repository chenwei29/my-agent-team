"""会话文件库（附件）服务。

元数据存 attachments 表，二进制存 `<workspace.root_path>/uploads/<id><ext>`。
所有文件路径都过 workspace 沙箱校验，绝不外溢。上传上限 20MB，空文件拒绝
（错误文案是 API 契约的一部分，前端把非 2xx body 当纯文本展示）。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from sqlalchemy import delete, desc, select

from app.db.models import Attachment, Workspace
from app.db.session import SessionLocal
from app.security.workspace_utils import is_path_within
from app.utils.ids import new_attachment_id
from app.utils.time import now_ms

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

_EXT_RE = re.compile(r"^\.[a-z0-9]{1,8}$")

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
}


def sanitize_ext(file_name: str) -> str:
    """合法扩展名（小写、`.`+1-8 位字母数字）原样返回，否则丢弃。"""
    ext = os.path.splitext(file_name)[1].lower()
    return ext if _EXT_RE.match(ext) else ""


def guess_mime(ext: str) -> str:
    return _MIME_BY_EXT.get(ext, "application/octet-stream")


async def upload_attachment(conversation_id: str, file_name: str, content: bytes, mime_hint: str) -> dict:
    """落盘 + 落库，返回 camelCase 的附件记录。"""
    if len(content) == 0:
        raise ValueError("Empty file")
    if len(content) > MAX_FILE_SIZE:
        raise ValueError(f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")

    from app.services.fs_service import get_workspace_for_conversation

    workspace = await get_workspace_for_conversation(conversation_id)
    if workspace is None:
        raise ValueError(f"Workspace not found for conversation: {conversation_id}")

    root = workspace.root_path
    attach_id = new_attachment_id()
    ext = sanitize_ext(file_name)
    stored_name = f"{attach_id}{ext}"
    abs_path = os.path.join(root, "uploads", stored_name)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    # 沙箱检查：词法解析后必须仍在 workspace 内
    if not is_path_within(os.path.abspath(abs_path), root):
        raise ValueError("Path traversal detected")

    with open(abs_path, "wb") as fh:
        fh.write(content)

    mime = mime_hint or guess_mime(ext)
    kind = "image" if mime.startswith("image/") else "file"

    row = Attachment(
        id=attach_id,
        conversation_id=conversation_id,
        kind=kind,
        file_name=file_name,
        file_path=f"uploads/{stored_name}",  # 相对 root 的 posix 风格路径
        size=len(content),
        mime_type=mime,
        created_at=now_ms(),
    )
    async with SessionLocal() as session:
        session.add(row)
        await session.commit()
        return _to_dict(row)


async def list_attachments(conversation_id: str) -> list[dict]:
    async with SessionLocal() as session:
        rows = (
            (
                await session.execute(
                    select(Attachment)
                    .where(Attachment.conversation_id == conversation_id)
                    .order_by(desc(Attachment.created_at))
                )
            )
            .scalars()
            .all()
        )
        return [_to_dict(r) for r in rows]


async def get_attachment(attachment_id: str) -> dict | None:
    async with SessionLocal() as session:
        row = await session.get(Attachment, attachment_id)
        return _to_dict(row) if row is not None else None


async def get_attachment_absolute_path(attachment_id: str) -> str | None:
    """返回附件文件的绝对路径；记录缺失、逃逸、文件不在盘上都返回 None。"""
    async with SessionLocal() as session:
        row = await session.get(Attachment, attachment_id)
        if row is None:
            return None
        workspace = await session.scalar(
            select(Workspace).where(Workspace.conversation_id == row.conversation_id)
        )
    if workspace is None:
        return None
    abs_path = os.path.join(workspace.root_path, row.file_path)
    if not is_path_within(os.path.abspath(abs_path), workspace.root_path):
        return None
    return abs_path if os.path.exists(abs_path) else None


async def delete_attachment(attachment_id: str) -> None:
    """先删库行，文件尽力删（失败不回滚——库里没了就算删成功）。"""
    abs_path = await get_attachment_absolute_path(attachment_id)
    async with SessionLocal() as session:
        result = await session.execute(
            delete(Attachment).where(Attachment.id == attachment_id)
        )
        await session.commit()
    if result.rowcount == 0:
        raise ValueError(f"Failed to delete attachment: {attachment_id}")
    if abs_path is not None:
        try:
            Path(abs_path).unlink(missing_ok=True)
        except OSError:
            pass


def _to_dict(row: Attachment) -> dict:
    return {
        "id": row.id,
        "conversationId": row.conversation_id,
        "kind": row.kind,
        "fileName": row.file_name,
        "filePath": row.file_path,
        "size": row.size,
        "mimeType": row.mime_type,
        "createdAt": row.created_at,
    }
