"""附件端点：会话内上传/列表 + 全局下载/删除。

上传走 multipart（字段名 `file`）；下载时图片 inline 预览、其余 attachment 下载，
Content-Disposition 用 `filename*=UTF-8''` 编码文件名。DELETE 任何失败都 404。
"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.errors import HttpError
from app.services.attachment_service import (
    delete_attachment,
    get_attachment,
    get_attachment_absolute_path,
    list_attachments,
    upload_attachment,
)

conv_router = APIRouter(prefix="/api/conversations/{conversation_id}/attachments", tags=["attachments"])
item_router = APIRouter(prefix="/api/attachments", tags=["attachments"])


@conv_router.get("")
async def list_conversation_attachments(conversation_id: str) -> dict:
    return {"attachments": await list_attachments(conversation_id)}


@conv_router.post("")
async def upload_conversation_attachment(conversation_id: str, file: UploadFile | None = None) -> dict:
    if file is None:
        raise HttpError(400, "Missing file")
    try:
        content = await file.read()
    except Exception as err:
        raise HttpError(400, f"Failed to read upload: {err}") from err

    try:
        attachment = await upload_attachment(
            conversation_id=conversation_id,
            file_name=file.filename or "",
            content=content,
            mime_hint=file.content_type or "",
        )
    except ValueError as err:  # 空文件 / 超大 / workspace 缺失等服务层错误一律 400
        raise HttpError(400, str(err)) from err
    return JSONResponse(status_code=201, content={"attachment": attachment})


@item_router.get("/{attachment_id}")
async def download_attachment(attachment_id: str):
    row = await get_attachment(attachment_id)
    if row is None:
        raise HttpError(404, "Not found")
    abs_path = await get_attachment_absolute_path(attachment_id)
    if abs_path is None:
        raise HttpError(410, "File missing on disk")

    disposition = "inline" if row["kind"] == "image" else "attachment"
    encoded_name = quote(row["fileName"])
    return FileResponse(
        abs_path,
        media_type=row["mimeType"],
        headers={
            "Content-Disposition": f"{disposition}; filename*=UTF-8''{encoded_name}",
            "Cache-Control": "private, max-age=3600",
        },
    )


@item_router.delete("/{attachment_id}")
async def remove_attachment(attachment_id: str) -> dict:
    try:
        await delete_attachment(attachment_id)
    except ValueError as err:
        raise HttpError(404, str(err)) from err
    return {"ok": True}
