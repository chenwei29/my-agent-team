"""read_attachment 工具：读「会话文件库」里的用户上传附件（id 前缀 `att_`）。

- id 以 `art_` 开头是产物（agent 自己产出的），提示改用 read_artifact
- 文本类（text/*、json/xml/js/yaml）直接读，超 50000 字符截断
- PDF 抽文本（pypdf），无文本时提示需要 OCR
- 图片只回元信息 + 说明（图片经多模态消息送达）
- 其他二进制只回元信息与格式说明

查询按会话隔离：别的会话传来的 attachmentId 视为不存在。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.attachment_service import (
    get_attachment,
    get_attachment_absolute_path,
)
from app.tools.types import ToolContext, ToolDef, ToolResult

MAX_TEXT_CHARS = 50_000

_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_FULL = {
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-yaml",
}


class ReadAttachmentArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attachment_id: str = Field(alias="attachmentId", min_length=1)


def _is_text_like(mime: str) -> bool:
    return mime in _TEXT_MIME_FULL or mime.startswith(_TEXT_MIME_PREFIXES)


def _is_pdf_like(mime: str, file_name: str, abs_path: str) -> bool:
    if mime == "application/pdf":
        return True
    if file_name.lower().endswith(".pdf"):
        return True
    try:
        with open(abs_path, "rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def _truncate(raw: str) -> tuple[str, bool]:
    truncated = len(raw) > MAX_TEXT_CHARS
    content = raw[:MAX_TEXT_CHARS] + f"\n\n[TRUNCATED at {MAX_TEXT_CHARS} chars]" if truncated else raw
    return content, truncated


def _extract_pdf_text(abs_path: str) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(abs_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    content, truncated = _truncate(text)
    result = {"content": content, "truncated": truncated, "pageCount": len(reader.pages)}
    if not text:
        result["note"] = (
            "No extractable text was found in this PDF. It may be scanned or image-only; "
            "OCR is required to inspect its content."
        )
    return result


async def _handle(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = ReadAttachmentArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid args: {err}")

    attachment_id = parsed.attachment_id

    # 防误用：artifact id 混进来时给更友好的提示
    if attachment_id.startswith("art_"):
        return ToolResult(
            ok=False,
            error=(
                f"'{attachment_id}' is an artifact id (art_*), not an attachment. "
                "Use read_artifact instead."
            ),
        )

    row = await get_attachment(attachment_id)
    if row is None or row["conversationId"] != ctx.conversation_id:
        return ToolResult(
            ok=False, error=f"Attachment not found in this conversation: {attachment_id}"
        )

    abs_path = await get_attachment_absolute_path(attachment_id)
    if abs_path is None:
        return ToolResult(ok=False, error="Attachment file missing on disk")

    meta = {
        "id": row["id"],
        "fileName": row["fileName"],
        "size": row["size"],
        "mimeType": row["mimeType"],
        "kind": row["kind"],
    }

    if _is_pdf_like(row["mimeType"], row["fileName"], abs_path):
        if ctx.abort_signal is not None and ctx.abort_signal.aborted:
            return ToolResult(ok=False, error="PDF extraction aborted")
        try:
            extracted = _extract_pdf_text(abs_path)
        except Exception as err:
            return ToolResult(ok=False, error=f"Failed to extract PDF text: {err}")
        if ctx.abort_signal is not None and ctx.abort_signal.aborted:
            return ToolResult(ok=False, error="PDF extraction aborted")
        return ToolResult(ok=True, value={**meta, **extracted})

    if _is_text_like(row["mimeType"]):
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as fh:
                content, truncated = _truncate(fh.read())
        except OSError as err:
            return ToolResult(ok=False, error=f"Failed to read text file: {err}")
        return ToolResult(ok=True, value={**meta, "content": content, "truncated": truncated})

    if row["kind"] == "image":
        return ToolResult(
            ok=True,
            value={
                **meta,
                "note": (
                    "Image bytes are delivered through the multimodal user message "
                    "(if the agent supports vision). You should already see this image "
                    "in the conversation content blocks."
                ),
            },
        )

    return ToolResult(
        ok=True,
        value={
            **meta,
            "note": (
                f"This is a {row['mimeType']} binary file. AgentHub does not yet extract "
                "text from this format; only metadata is available. Ask the user for a "
                "text version if you need to inspect content."
            ),
        },
    )


READ_ATTACHMENT_TOOL = ToolDef(
    name="read_attachment",
    description=(
        "Read the contents of a user-uploaded attachment (id starts with 'att_'). "
        "Use this when the user prompt mentions [图片附件: ...] or [文件附件: ...]. "
        "Returns plain text for text-like files (txt/md/json/csv/etc) and extractable "
        "PDF text. For images and unsupported binary formats only metadata is returned. "
        "Do NOT use this for ids starting with 'art_' — that's for read_artifact."
    ),
    parameters={
        "type": "object",
        "required": ["attachmentId"],
        "properties": {
            "attachmentId": {
                "type": "string",
                "description": "Attachment id, format att_xxx (NOT an artifact id)",
            },
        },
    },
    handler=_handle,
)
