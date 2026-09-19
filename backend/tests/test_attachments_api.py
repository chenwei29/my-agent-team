"""附件 API：上传 / 列表 / 下载 / 删除 + read_attachment 工具。"""

from __future__ import annotations

import os

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.db.models import Workspace
from app.db.session import SessionLocal
from app.tools.read_attachment import READ_ATTACHMENT_TOOL
from app.tools.registry import tool_registry
from app.tools.types import ToolContext


async def _workspace_root(conversation_id: str) -> str:
    async with SessionLocal() as session:
        ws = await session.scalar(select(Workspace).where(Workspace.conversation_id == conversation_id))
    assert ws is not None
    return ws.root_path


def _ctx(conversation_id: str) -> ToolContext:
    return ToolContext(
        conversation_id=conversation_id,
        agent_id="ag_test",
        run_id="run_test",
        workspace_path=".",
        abort_signal=None,
    )


async def _upload(client: AsyncClient, conv_id: str, name: str, data: bytes, mime: str = ""):
    files = {"file": (name, data, mime)} if mime else {"file": (name, data)}
    return await client.post(f"/api/conversations/{conv_id}/attachments", files=files)


async def test_upload_and_list_roundtrip(client: AsyncClient, conversation: dict):
    res = await _upload(client, conversation["id"], "笔记.md", "你好".encode("utf-8"), "text/markdown")
    assert res.status_code == 201, res.text
    att = res.json()["attachment"]
    assert att["fileName"] == "笔记.md"
    assert att["kind"] == "file"
    assert att["size"] == len("你好".encode("utf-8"))
    assert att["filePath"].startswith("uploads/att_")
    assert att["id"].startswith("att_")

    # 文件真的落在 workspace uploads/ 下
    root = await _workspace_root(conversation["id"])
    assert os.path.isfile(os.path.join(root, att["filePath"]))

    res = await client.get(f"/api/conversations/{conversation['id']}/attachments")
    assert res.status_code == 200
    assert [a["id"] for a in res.json()["attachments"]] == [att["id"]]


async def test_upload_image_kind_and_mime_guess(client: AsyncClient, conversation: dict):
    # mime 优先用上传值：非 image mime 即使扩展名是 png 也归为 file
    res = await _upload(client, conversation["id"], "pic.png", b"\x89PNG\r\n\x1a\n", "application/json")
    assert res.status_code == 201, res.text
    att = res.json()["attachment"]
    assert att["mimeType"] == "application/json"
    assert att["kind"] == "file"

    # mime 为空时按扩展名猜
    res = await _upload(client, conversation["id"], "pic.png", b"\x89PNG\r\n\x1a\n", "image/png")
    att = res.json()["attachment"]
    assert att["kind"] == "image"


def test_mime_guess_unit():
    from app.services.attachment_service import guess_mime, sanitize_ext

    assert guess_mime(".png") == "image/png"
    assert guess_mime(".pdf") == "application/pdf"
    assert guess_mime(".unknown") == "application/octet-stream"
    assert sanitize_ext("a.tar.gz") == ".gz"  # 只看最后一段
    assert sanitize_ext("noext") == ""
    assert sanitize_ext("file.toolongext") == ""  # 超过 8 位
    assert sanitize_ext("file.UPPER") == ".upper"


async def test_upload_empty_file_rejected(client: AsyncClient, conversation: dict):
    res = await _upload(client, conversation["id"], "empty.txt", b"")
    assert res.status_code == 400
    assert res.json()["error"] == "Empty file"


async def test_upload_too_large_rejected(client: AsyncClient, conversation: dict):
    res = await _upload(client, conversation["id"], "big.bin", b"x" * (20 * 1024 * 1024 + 1))
    assert res.status_code == 400
    assert res.json()["error"] == "File too large (max 20MB)"


async def test_upload_weird_extension_stripped(client: AsyncClient, conversation: dict):
    # 超长扩展名非法 → 丢弃，无后缀存储
    res = await _upload(client, conversation["id"], "data.weirdlongext", b"xx", "text/plain")
    assert res.status_code == 201
    att = res.json()["attachment"]
    assert att["filePath"] == f"uploads/{att['id']}"


async def test_download_inline_for_image_and_410_when_missing(
    client: AsyncClient, conversation: dict
):
    res = await _upload(client, conversation["id"], "a.png", b"\x89PNG", "image/png")
    att = res.json()["attachment"]

    res = await client.get(f"/api/attachments/{att['id']}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("image/png")
    assert res.headers["content-disposition"].startswith("inline;")
    assert "filename*=UTF-8''" in res.headers["content-disposition"]
    assert res.headers["cache-control"] == "private, max-age=3600"
    assert res.content == b"\x89PNG"

    # 非图片走 attachment 下载
    res = await _upload(client, conversation["id"], "a.txt", b"hi", "text/plain")
    res = await client.get(f"/api/attachments/{res.json()['attachment']['id']}")
    assert res.headers["content-disposition"].startswith("attachment;")

    # 盘上文件没了 → 410
    root = await _workspace_root(conversation["id"])
    os.remove(os.path.join(root, att["filePath"]))
    res = await client.get(f"/api/attachments/{att['id']}")
    assert res.status_code == 410
    assert res.json()["error"] == "File missing on disk"


async def test_download_unknown_404(client: AsyncClient):
    res = await client.get("/api/attachments/att_nope")
    assert res.status_code == 404
    assert res.json()["error"] == "Not found"


async def test_delete_attachment(client: AsyncClient, conversation: dict):
    res = await _upload(client, conversation["id"], "gone.txt", b"bye", "text/plain")
    att = res.json()["attachment"]
    root = await _workspace_root(conversation["id"])

    res = await client.delete(f"/api/attachments/{att['id']}")
    assert res.status_code == 200
    assert res.json() == {"ok": True}
    assert not os.path.exists(os.path.join(root, att["filePath"]))
    assert (await client.get(f"/api/conversations/{conversation['id']}/attachments")).json()[
        "attachments"
    ] == []

    # 重复删 → 404；未知 id → 404
    assert (await client.delete(f"/api/attachments/{att['id']}")).status_code == 404
    assert (await client.delete("/api/attachments/att_nope")).status_code == 404


async def test_list_ordered_desc(client: AsyncClient, conversation: dict):
    ids = []
    for i in range(3):
        res = await _upload(client, conversation["id"], f"f{i}.txt", str(i).encode(), "text/plain")
        ids.append(res.json()["attachment"]["id"])
    res = await client.get(f"/api/conversations/{conversation['id']}/attachments")
    assert [a["id"] for a in res.json()["attachments"]] == list(reversed(ids))


# ─── read_attachment 工具 ─────────────────────────────────


@pytest.fixture(autouse=True)
def _register_tool():
    tool_registry.register(READ_ATTACHMENT_TOOL)
    yield
    tool_registry._tools.pop(READ_ATTACHMENT_TOOL.name, None)


async def test_read_attachment_text_truncated(client: AsyncClient, conversation: dict):
    long_text = "x" * 60_000
    res = await _upload(client, conversation["id"], "long.txt", long_text.encode(), "text/plain")
    att = res.json()["attachment"]

    result = await READ_ATTACHMENT_TOOL.handler({"attachmentId": att["id"]}, _ctx(conversation["id"]))
    assert result.ok
    assert result.value["content"].startswith("xxxx")
    assert result.value["truncated"] is True
    assert result.value["content"].endswith("[TRUNCATED at 50000 chars]")
    assert result.value["fileName"] == "long.txt"


async def test_read_attachment_scoped_to_conversation(client: AsyncClient, conversation: dict):
    res = await _upload(client, conversation["id"], "mine.txt", b"hi", "text/plain")
    att = res.json()["attachment"]

    result = await READ_ATTACHMENT_TOOL.handler(
        {"attachmentId": att["id"]}, _ctx("conv_someone_else")
    )
    assert not result.ok
    assert "not found in this conversation" in result.error


async def test_read_attachment_artifact_id_hint(client: AsyncClient):
    result = await READ_ATTACHMENT_TOOL.handler({"attachmentId": "art_123"}, _ctx("conv_x"))
    assert not result.ok
    assert "read_artifact" in result.error


async def test_read_attachment_image_note(client: AsyncClient, conversation: dict):
    res = await _upload(client, conversation["id"], "img.png", b"\x89PNG", "image/png")
    att = res.json()["attachment"]

    result = await READ_ATTACHMENT_TOOL.handler({"attachmentId": att["id"]}, _ctx(conversation["id"]))
    assert result.ok
    assert "multimodal" in result.value["note"]
    assert "content" not in result.value


async def test_read_attachment_binary_note(client: AsyncClient, conversation: dict):
    res = await _upload(client, conversation["id"], "doc.zip", b"PK\x03\x04", "application/zip")
    att = res.json()["attachment"]

    result = await READ_ATTACHMENT_TOOL.handler({"attachmentId": att["id"]}, _ctx(conversation["id"]))
    assert result.ok
    assert "application/zip" in result.value["note"]


async def test_read_attachment_pdf(client: AsyncClient, conversation: dict):
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # 空白页没有可抽文本 → OCR 提示
    import io

    buf = io.BytesIO()
    writer.write(buf)
    res = await _upload(client, conversation["id"], "doc.pdf", buf.getvalue(), "application/pdf")
    att = res.json()["attachment"]

    result = await READ_ATTACHMENT_TOOL.handler({"attachmentId": att["id"]}, _ctx(conversation["id"]))
    assert result.ok
    assert result.value["pageCount"] == 1
    assert "OCR" in result.value["note"]
