"""产物端点：列表（带会话标题）/ 详情 / 版本链 / 删除 / 预览 / 导出。"""

from __future__ import annotations

import io
import zipfile

import pytest
from httpx import AsyncClient

from tests.test_tools_artifacts import _ctx  # 复用同一个 ToolContext 构造器

MOCK_ID = "ag_e2e_mock"


@pytest.fixture
async def conversation(client: AsyncClient) -> dict:
    res = await client.post("/api/conversations", json={"mode": "single", "agentIds": [MOCK_ID]})
    assert res.status_code == 201, res.text
    return res.json()["conversation"]


async def _write(client: AsyncClient, conversation_id: str, args: dict) -> str:
    from app.tools.artifacts import WRITE_ARTIFACT_TOOL

    result = await WRITE_ARTIFACT_TOOL.handler(args, _ctx(conversation_id, MOCK_ID))
    assert result.ok, result.error
    return result.value["artifactId"]


async def test_list_includes_conversation_title(client: AsyncClient, conversation: dict):
    artifact_id = await _write(
        client,
        conversation["id"],
        {
            "type": "document",
            "title": "列表项",
            "content": {"format": "markdown", "content": "x"},
        },
    )
    res = await client.get("/api/artifacts")
    assert res.status_code == 200
    items = res.json()["artifacts"]
    mine = next(a for a in items if a["id"] == artifact_id)
    assert mine["conversationId"] == conversation["id"]
    assert mine["conversationTitle"] == conversation["title"]
    assert set(mine.keys()) == {
        "id",
        "conversationId",
        "conversationTitle",
        "type",
        "title",
        "version",
        "parentArtifactId",
        "createdByAgentId",
        "createdAt",
    }


async def test_get_artifact(client: AsyncClient, conversation: dict):
    artifact_id = await _write(
        client,
        conversation["id"],
        {
            "type": "document",
            "title": "详情",
            "content": {"format": "markdown", "content": "body"},
        },
    )
    res = await client.get(f"/api/artifacts/{artifact_id}")
    assert res.status_code == 200
    artifact = res.json()["artifact"]
    assert artifact["id"] == artifact_id
    assert artifact["title"] == "详情"
    assert artifact["version"] == 1
    assert artifact["createdAt"] > 0

    res = await client.get("/api/artifacts/art_missing")
    assert res.status_code == 404
    assert res.json() == {"error": "Not found"}


async def test_delete_artifact(client: AsyncClient, conversation: dict):
    artifact_id = await _write(
        client,
        conversation["id"],
        {
            "type": "document",
            "title": "待删",
            "content": {"format": "markdown", "content": "x"},
        },
    )
    res = await client.delete(f"/api/artifacts/{artifact_id}")
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    res = await client.delete(f"/api/artifacts/{artifact_id}")
    assert res.status_code == 404
    assert "not found" in res.json()["error"]


async def test_create_version_and_chain(client: AsyncClient, conversation: dict):
    v1 = await _write(
        client,
        conversation["id"],
        {"type": "document", "title": "v1", "content": {"format": "markdown", "content": "a"}},
    )

    res = await client.post(
        f"/api/artifacts/{v1}/versions",
        json={"content": {"format": "markdown", "content": "b"}, "title": "  v2 名  "},
    )
    assert res.status_code == 200, res.text
    v2 = res.json()["artifact"]
    assert v2["version"] == 2
    assert v2["parentArtifactId"] == v1
    assert v2["title"] == "v2 名"
    assert v2["content"]["content"] == "b"

    # 版本链：从 v2 查也是整条链，升序
    res = await client.get(f"/api/artifacts/{v2['id']}/versions")
    assert res.status_code == 200
    versions = res.json()["versions"]
    assert [row["version"] for row in versions] == [1, 2]

    # 从 v1 查同一条链
    res = await client.get(f"/api/artifacts/{v1}/versions")
    assert [row["version"] for row in res.json()["versions"]] == [1, 2]

    # 不继承 title 时沿用 parent 的
    res = await client.post(
        f"/api/artifacts/{v1}/versions",
        json={"content": {"format": "markdown", "content": "c"}},
    )
    assert res.json()["artifact"]["title"] == "v1"


async def test_create_version_errors(client: AsyncClient, conversation: dict):
    res = await client.post(
        "/api/artifacts/art_nope/versions",
        json={"content": {"format": "markdown", "content": "x"}},
    )
    assert res.status_code == 404
    assert "not found" in res.json()["error"]

    artifact_id = await _write(
        client,
        conversation["id"],
        {
            "type": "diagram",
            "title": "图",
            "content": {"syntax": "mermaid", "source": "flowchart TD\nA-->B"},
        },
    )
    res = await client.post(
        f"/api/artifacts/{artifact_id}/versions",
        json={"content": {"syntax": "mermaid", "source": "not a mermaid"}},
    )
    assert res.status_code == 400
    assert res.json()["error"].startswith("Invalid Mermaid diagram:")

    res = await client.get("/api/artifacts/art_nope/versions")
    assert res.status_code == 404


async def test_preview_web_app(client: AsyncClient, conversation: dict):
    artifact_id = await _write(
        client,
        conversation["id"],
        {
            "type": "web_app",
            "title": "页面",
            "content": {
                "files": {
                    "index.html": "<html><head></head><body><h1>hi</h1></body></html>",
                    "style.css": "h1{color:red}",
                    "script.js": "console.log(1)",
                }
            },
        },
    )
    res = await client.get(f"/api/artifacts/{artifact_id}/preview")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "sandbox allow-scripts" in res.headers["content-security-policy"]
    assert "<style>" in res.text and "<script>" in res.text
    assert res.headers["cache-control"] == "no-store"

    doc_id = await _write(
        client,
        conversation["id"],
        {"type": "document", "title": "文档", "content": {"format": "markdown", "content": "x"}},
    )
    res = await client.get(f"/api/artifacts/{doc_id}/preview")
    assert res.status_code == 400
    assert res.json() == {"error": "Artifact is not a web_app"}


async def test_export_variants(client: AsyncClient, conversation: dict):
    web_id = await _write(
        client,
        conversation["id"],
        {
            "type": "web_app",
            "title": "导出 应用",
            "content": {"files": {"index.html": "<h1>x</h1>"}},
        },
    )
    res = await client.get(f"/api/artifacts/{web_id}/export")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/zip"
    assert 'filename="' in res.headers["content-disposition"]
    archive = zipfile.ZipFile(io.BytesIO(res.content))
    assert set(archive.namelist()) == {"index.html", "README.txt"}
    assert archive.read("index.html") == b"<h1>x</h1>"
    assert b"Entry: index.html" in archive.read("README.txt")

    doc_id = await _write(
        client,
        conversation["id"],
        {"type": "document", "title": "文档", "content": {"format": "markdown", "content": "# t"}},
    )
    res = await client.get(f"/api/artifacts/{doc_id}/export")
    assert res.status_code == 200
    assert res.headers["content-type"] == "text/markdown; charset=utf-8"
    assert res.text == "# t"

    img_id = await _write(
        client,
        conversation["id"],
        {"type": "image", "title": "图", "content": {"url": "https://example.com/a.png"}},
    )
    res = await client.get(f"/api/artifacts/{img_id}/export")
    assert res.status_code == 302
    assert res.headers["location"] == "https://example.com/a.png"

    res = await client.get(f"/api/artifacts/{doc_id}/export?mode=bogus")
    assert res.status_code == 400
    assert res.json()["error"] == "Unsupported export mode: bogus"
