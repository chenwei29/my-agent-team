"""部署：候选列表 / 部署端点 / 静态文件服务 / 下载包 / 外部静态发布。"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.db.models import Conversation, Message, Workspace
from app.db.session import SessionLocal
from app.services import deployment_service
from app.tools.deploy import DEPLOY_ARTIFACT_TOOL, DEPLOY_WORKSPACE_TOOL
from tests.test_tools_artifacts import _ctx

MOCK_ID = "ag_e2e_mock"

WEB_APP_ARGS = {
    "type": "web_app",
    "title": "待部署页面",
    "content": {
        "files": {
            "index.html": "<html><head></head><body><h1>hi</h1></body></html>",
            "style.css": "h1 { color: red }",
            "script.js": "console.log('ready')",
        },
        "entry": "index.html",
    },
}


async def _write_artifact(conversation_id: str, args: dict = WEB_APP_ARGS) -> str:
    from app.tools.artifacts import WRITE_ARTIFACT_TOOL

    result = await WRITE_ARTIFACT_TOOL.handler(args, _ctx(conversation_id, MOCK_ID))
    assert result.ok, result.error
    return result.value["artifactId"]


async def _deploy(client: AsyncClient, conversation_id: str, artifact_id: str | None) -> dict:
    body = {"artifactId": artifact_id} if artifact_id else {}
    res = await client.post(f"/api/conversations/{conversation_id}/deploy", json=body)
    assert res.status_code == 200, res.text
    return res.json()


async def _rows(conversation_id: str) -> list[Message]:
    async with SessionLocal() as session:
        return list(
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
            )
        )


# ─── 部署端点 ──────────────────────────────────────────────


async def test_deploy_artifact_creates_deployment_and_system_message(
    client: AsyncClient, conversation: dict
):
    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)

    result = await _deploy(client, cid, artifact_id)
    assert result["kind"] == "deployed"

    deployment = result["deployment"]
    assert deployment["status"] == "ready"
    assert deployment["artifactId"] == artifact_id
    assert deployment["title"] == "待部署页面"
    assert deployment["version"] == 1
    assert deployment["sourceType"] == "artifact"
    assert deployment["deploymentType"] == "local_static"
    assert deployment["previewPath"] == f"/api/deployments/{deployment['id']}"
    assert deployment["sourceDownloadPath"].endswith("/download/source")
    assert deployment["containerDownloadPath"].endswith("/download/container")

    # 系统消息带 deploy_status part，并顶起会话 updatedAt
    messages = await _rows(cid)
    assert [m.role for m in messages] == ["system"]
    assert messages[0].parts == [{"type": "deploy_status", "deployment": deployment}]
    assert result["message"]["id"] == messages[0].id

    async with SessionLocal() as session:
        conv = await session.get(Conversation, cid)
    assert conv.updated_at >= messages[0].created_at


async def test_deploy_candidates_endpoint(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)
    # document 不是候选
    await _write_artifact(
        cid,
        {"type": "document", "title": "文档", "content": {"format": "markdown", "content": "x"}},
    )

    res = await client.get(f"/api/conversations/{cid}/deploy")
    assert res.status_code == 200
    candidates = res.json()["candidates"]
    assert candidates == [
        {
            "artifactId": artifact_id,
            "title": "待部署页面",
            "version": 1,
            "createdByAgentId": MOCK_ID,
            "createdAt": candidates[0]["createdAt"],
        }
    ]


async def test_deploy_without_candidates(client: AsyncClient, conversation: dict):
    result = await _deploy(client, conversation["id"], None)
    assert result["kind"] == "no_candidates"
    assert result["candidates"] == []
    assert result["message"]["parts"][0]["type"] == "text"
    assert "还没有可部署的网页产物" in result["message"]["parts"][0]["content"]


async def test_deploy_single_candidate_is_automatic(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)
    result = await _deploy(client, cid, None)
    assert result["kind"] == "deployed"
    assert result["deployment"]["artifactId"] == artifact_id


async def test_deploy_multiple_candidates_returns_selection(
    client: AsyncClient, conversation: dict
):
    cid = conversation["id"]
    first = await _write_artifact(cid)
    second = await _write_artifact(cid, {**WEB_APP_ARGS, "title": "第二个页面"})

    result = await _deploy(client, cid, None)
    assert result["kind"] == "candidate_selection"
    assert {c["artifactId"] for c in result["candidates"]} == {first, second}
    # 候选按 createdAt 倒序（同毫秒建的两条之间不保证先后）
    created = [c["createdAt"] for c in result["candidates"]]
    assert created == sorted(created, reverse=True)
    assert result["message"]["parts"][0]["type"] == "deploy_candidates"


async def test_deploy_non_web_app_artifact_fails(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    artifact_id = await _write_artifact(
        cid,
        {"type": "document", "title": "文档", "content": {"format": "markdown", "content": "x"}},
    )

    result = await _deploy(client, cid, artifact_id)
    assert result["kind"] == "deployed"
    deployment = result["deployment"]
    assert deployment["status"] == "failed"
    assert deployment["error"] == 'Artifact type "document" cannot be deployed as a web app'
    assert deployment["previewPath"] == f"/api/artifacts/{artifact_id}/preview"


async def test_deploy_unknown_artifact_fails(client: AsyncClient, conversation: dict):
    result = await _deploy(client, conversation["id"], "art_missing")
    assert result["deployment"]["status"] == "failed"
    assert result["deployment"]["error"] == "Artifact not found"


async def test_deploy_command_message_skips_runs(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)

    res = await client.post(f"/api/conversations/{cid}/messages", json={"content": "部署"})
    assert res.status_code == 202
    body = res.json()
    assert body["runIds"] == []
    assert body["deploy"]["kind"] == "deployed"
    assert body["deploy"]["deployment"]["artifactId"] == artifact_id
    assert len(body["messages"]) == 1

    # 带产物 id 的指令同样识别
    res = await client.post(
        f"/api/conversations/{cid}/messages", json={"content": f"/deploy {artifact_id}"}
    )
    assert res.json()["deploy"]["deployment"]["artifactId"] == artifact_id

    # 普通提问不会被当成指令（deploy 字段在响应里直接不出现，对应前端 `deploy?`）
    res = await client.post(
        f"/api/conversations/{cid}/messages", json={"content": "帮我部署一下这个页面可以吗"}
    )
    assert "deploy" not in res.json()


# ─── 工作区部署 ────────────────────────────────────────────


async def _workspace_cwd(conversation_id: str) -> Path:
    async with SessionLocal() as session:
        workspace = await session.scalar(
            select(Workspace).where(Workspace.conversation_id == conversation_id)
        )
    return Path(workspace.root_path)


async def test_deploy_workspace_tool(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    cwd = await _workspace_cwd(cid)
    (cwd / "dist").mkdir(parents=True, exist_ok=True)
    (cwd / "dist" / "index.html").write_text("<html><body>built</body></html>")
    (cwd / "dist" / "app.js").write_text("console.log(1)")

    result = await DEPLOY_WORKSPACE_TOOL.handler({"path": "dist"}, _ctx(cid, MOCK_ID))
    assert result.ok
    deployment = result.value
    assert deployment["status"] == "ready"
    assert deployment["sourceType"] == "workspace"
    assert deployment["workspacePath"] == "dist"
    assert deployment["title"] == "Workspace dist"
    assert deployment["artifactId"] == "workspace:dist"
    assert deployment["version"] == 0

    res = await client.get(deployment["previewPath"])
    assert res.status_code == 200
    assert b"built" in res.content

    # 目录里有 index.html 时，「部署」指令会自动挑中它
    assert (await _deploy(client, cid, None))["deployment"]["workspacePath"] == "dist"


async def test_deploy_workspace_outside_workspace_fails(
    client: AsyncClient, conversation: dict
):
    result = await DEPLOY_WORKSPACE_TOOL.handler({"path": "../escape"}, _ctx(conversation["id"], MOCK_ID))
    assert result.ok
    assert result.value["status"] == "failed"
    assert "outside workspace" in result.value["error"]


async def test_deploy_artifact_tool(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)
    result = await DEPLOY_ARTIFACT_TOOL.handler({"artifactId": artifact_id}, _ctx(cid, MOCK_ID))
    assert result.ok
    assert result.value["status"] == "ready"
    assert result.value["artifactId"] == artifact_id

    missing = await DEPLOY_ARTIFACT_TOOL.handler({"artifactId": "art_nope"}, _ctx(cid, MOCK_ID))
    assert missing.value["status"] == "failed"


# ─── 静态文件服务 ──────────────────────────────────────────


async def test_deployment_static_assets(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)
    deployment = (await _deploy(client, cid, artifact_id))["deployment"]
    base = deployment["previewPath"]

    # 根路径给运行入口：iframe 骨架，把 css/js 注进 head
    res = await client.get(base)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert res.headers["x-content-type-options"] == "nosniff"
    assert "sandbox allow-scripts" in res.headers["content-security-policy"]
    body = res.text
    assert "h1 { color: red }" in body
    assert "console.log('ready')" in body
    assert "<h1>hi</h1>" in body

    # 产物原始文件按扩展名给 MIME
    res = await client.get(f"{base}/style.css")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/css")
    assert res.text == "h1 { color: red }"

    res = await client.get(f"{base}/script.js")
    assert res.headers["content-type"].startswith("text/javascript")

    # index.html 就是拼好的运行入口（原始文件留在 .agenthub/source/ 里，不对外）
    res = await client.get(f"{base}/index.html")
    assert res.text == body
    assert (
        deployment_service.deployment_dir(deployment["id"]) / ".agenthub" / "source" / "index.html"
    ).read_text() == WEB_APP_ARGS["content"]["files"]["index.html"]


async def test_deployment_asset_errors(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)
    deployment = (await _deploy(client, cid, artifact_id))["deployment"]
    base = deployment["previewPath"]

    assert (await client.get(f"{base}/missing.css")).status_code == 404
    # 私有目录一律当作不存在
    assert (await client.get(f"{base}/.agenthub/manifest.json")).status_code == 404
    assert (await client.get(f"{base}/.agenthub/source/index.html")).status_code == 404
    # 穿越与非法 path
    assert (await client.get(f"{base}/..%2F..%2Fetc%2Fpasswd")).status_code in (400, 404)
    # 目录穿越的规范化形态
    assert (await client.get(f"{base}/foo/../../.agenthub/manifest.json")).status_code == 404
    # 不存在的部署 / 非法 id
    assert (await client.get("/api/deployments/dep_missing")).status_code == 404
    assert (await client.get("/api/deployments/nope")).status_code == 404


# ─── 下载包 ────────────────────────────────────────────────


async def test_deployment_downloads(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)
    deployment = (await _deploy(client, cid, artifact_id))["deployment"]

    res = await client.get(deployment["sourceDownloadPath"])
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(res.content)) as archive:
        names = set(archive.namelist())
        assert {"index.html", "style.css", "script.js", "README.txt"} <= names
        assert archive.read("style.css").decode() == "h1 { color: red }"
        # 源码包只放原始文件，不放生成出来的运行入口
        assert ".agenthub/manifest.json" not in names

    res = await client.get(deployment["containerDownloadPath"])
    assert res.status_code == 200
    with zipfile.ZipFile(io.BytesIO(res.content)) as archive:
        names = set(archive.namelist())
        assert {"Dockerfile", "nginx.conf", "README.txt", "app/index.html"} <= names
        assert "nginx:1.27-alpine" in archive.read("Dockerfile").decode()

    assert (await client.get(f"/api/deployments/{deployment['id']}/download/nope")).status_code == 400
    assert (await client.get("/api/deployments/dep_missing/download/source")).status_code == 404


# ─── 部署目录布局 ──────────────────────────────────────────


async def test_deployment_directory_layout(conversation: dict):
    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)
    deployment = (
        await DEPLOY_ARTIFACT_TOOL.handler({"artifactId": artifact_id}, _ctx(cid, MOCK_ID))
    ).value

    root = deployment_service.deployment_dir(deployment["id"])
    assert (root / "index.html").is_file()
    assert (root / "style.css").is_file()
    assert (root / ".agenthub" / "source" / "index.html").is_file()

    manifest = json.loads((root / ".agenthub" / "manifest.json").read_text())
    assert manifest["sourceFiles"] == ["index.html", "script.js", "style.css"]
    assert manifest["runtimeEntry"] == "index.html"
    assert manifest["deploymentType"] == "local_static"
    assert deployment_service.read_deployment_manifest("dep_missing") is None
    assert deployment_service.read_deployment_manifest("nope") is None


async def test_duplicate_deployment_id_rejected(client: AsyncClient, conversation: dict):
    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)
    deployment = (await _deploy(client, cid, artifact_id))["deployment"]

    with pytest.raises(deployment_service.DeploymentError, match="already exists"):
        deployment_service.create_local_static_deployment(
            deployment_id=deployment["id"],
            artifact_id=artifact_id,
            title="再来一次",
            version=1,
            content=WEB_APP_ARGS["content"],
            created_at=0,
        )


async def test_deployment_rejects_unsafe_file_paths(client: AsyncClient, conversation: dict):
    with pytest.raises(deployment_service.DeploymentError, match="Unsafe web app file path"):
        deployment_service.create_local_static_deployment(
            deployment_id="dep_safecheck",
            artifact_id="art_x",
            title="坏路径",
            version=1,
            content={"files": {"../escape.html": "<html></html>"}, "entry": "index.html"},
            created_at=0,
        )
    # 半成品目录不会残留，否则同名 id 重试会被「已存在」挡住
    assert not deployment_service.deployment_dir("dep_safecheck").exists()


def test_normalize_deployment_file_path():
    normalize = deployment_service.normalize_deployment_file_path
    assert normalize("assets/app.js") == "assets/app.js"
    assert normalize(r"assets\app.js") == "assets/app.js"
    # 词法归一：`.` 段被折叠，但结果仍在根目录内
    assert normalize("./a") == "a"
    for bad in ("", "  ", "/etc/passwd", "C:/windows", "a/../b", "a//b", ".agenthub/x", ".."):
        assert normalize(bad) is None, bad


# ─── 外部静态发布 ──────────────────────────────────────────


@pytest.fixture
async def publish_target(tmp_path) -> str:
    target = tmp_path / "published"
    target.mkdir()
    return str(target)


@pytest.fixture(autouse=True)
async def _reset_publish_settings(client: AsyncClient):
    """整个 DB 是测试会话共享的，发布配置必须用完就清，免得污染后续用例。"""
    yield
    await client.patch(
        "/api/settings",
        json={
            "deploymentPublishEnabled": False,
            "deploymentPublishDir": None,
            "deploymentPublicBaseUrl": None,
        },
    )


async def test_external_publish_moves_preview_to_public_url(
    client: AsyncClient, conversation: dict, publish_target: str
):
    res = await client.patch(
        "/api/settings",
        json={
            "deploymentPublishEnabled": True,
            "deploymentPublishDir": publish_target,
            "deploymentPublicBaseUrl": "https://cdn.example.com/site",
        },
    )
    assert res.status_code == 200, res.text

    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)
    deployment = (await _deploy(client, cid, artifact_id))["deployment"]

    assert deployment["status"] == "ready"
    assert deployment["deploymentType"] == "external_static"
    assert deployment["previewPath"] == f"https://cdn.example.com/site/{deployment['id']}/"
    assert deployment["publicUrl"] == deployment["previewPath"]
    assert deployment["localPreviewPath"] == f"/api/deployments/{deployment['id']}"
    assert deployment["publishTargetType"] == "static_directory"

    # 公开文件被拷过去，私有目录不拷
    published = Path(publish_target) / deployment["id"]
    assert (published / "index.html").is_file()
    assert not (published / ".agenthub").exists()
    # 本地预览仍然可用
    assert (await client.get(deployment["localPreviewPath"])).status_code == 200


async def test_external_publish_without_config_fails(
    client: AsyncClient, conversation: dict
):
    res = await client.patch(
        "/api/settings",
        json={
            "deploymentPublishEnabled": True,
            "deploymentPublishDir": None,
            "deploymentPublicBaseUrl": None,
        },
    )
    assert res.status_code == 200, res.text

    cid = conversation["id"]
    artifact_id = await _write_artifact(cid)
    deployment = (await _deploy(client, cid, artifact_id))["deployment"]

    assert deployment["status"] == "failed"
    assert deployment["deploymentType"] == "external_static"
    assert "publish directory or public base URL is not configured" in deployment["error"]
    assert deployment["localPreviewPath"] == f"/api/deployments/{deployment['id']}"
