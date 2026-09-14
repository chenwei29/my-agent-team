"""`/api/settings`、`/api/fs/listdir`、`/api/platform` 契约测试。"""

from __future__ import annotations

import os
from pathlib import Path

from httpx import AsyncClient

SETTINGS_CAMEL_KEYS = {
    "anthropicApiKey",
    "anthropicBaseUrl",
    "openaiApiKey",
    "deepseekApiKey",
    "arkApiKey",
    "deploymentPublishEnabled",
    "deploymentPublishDir",
    "deploymentPublicBaseUrl",
    "updatedAt",
}


# ─── /api/platform ──────────────────────────────────────
async def test_platform_reports_posix(client: AsyncClient):
    res = await client.get("/api/platform")
    assert res.status_code == 200
    assert res.json() == {"platform": "posix"}


# ─── /api/settings ──────────────────────────────────────
async def test_settings_defaults_before_any_write(client: AsyncClient):
    settings = (await client.get("/api/settings")).json()["settings"]
    assert settings["id"] == "singleton"
    assert settings["anthropicApiKey"] is None
    assert settings["deploymentPublishEnabled"] is False
    assert SETTINGS_CAMEL_KEYS <= set(settings)


async def test_settings_patch_then_get(client: AsyncClient):
    res = await client.patch("/api/settings", json={"deepseekApiKey": "sk-test-123"})
    assert res.status_code == 200, res.text
    assert res.json()["settings"]["deepseekApiKey"] == "sk-test-123"

    # 再读一次：真的落库了
    settings = (await client.get("/api/settings")).json()["settings"]
    assert settings["deepseekApiKey"] == "sk-test-123"
    assert settings["updatedAt"] > 0

    await client.patch("/api/settings", json={"deepseekApiKey": None})


async def test_settings_patch_only_touches_provided_fields(client: AsyncClient):
    await client.patch("/api/settings", json={"openaiApiKey": "sk-openai"})
    res = await client.patch("/api/settings", json={"arkApiKey": "sk-ark"})
    body = res.json()["settings"]
    # 上一次写的没被这次覆盖成 null
    assert body["openaiApiKey"] == "sk-openai"
    assert body["arkApiKey"] == "sk-ark"

    await client.patch("/api/settings", json={"openaiApiKey": None, "arkApiKey": None})


async def test_settings_normalizes_blank_string_to_null(client: AsyncClient):
    await client.patch("/api/settings", json={"anthropicApiKey": "  sk-x  "})
    assert (await client.get("/api/settings")).json()["settings"]["anthropicApiKey"] == "sk-x"

    res = await client.patch("/api/settings", json={"anthropicApiKey": "   "})
    assert res.json()["settings"]["anthropicApiKey"] is None


async def test_settings_patch_boolean(client: AsyncClient):
    res = await client.patch("/api/settings", json={"deploymentPublishEnabled": True})
    assert res.json()["settings"]["deploymentPublishEnabled"] is True

    await client.patch("/api/settings", json={"deploymentPublishEnabled": False})


# ─── /api/fs/listdir ────────────────────────────────────
async def test_listdir_defaults_to_home(client: AsyncClient):
    res = await client.get("/api/fs/listdir")
    assert res.status_code == 200
    body = res.json()
    assert body["path"] == str(Path.home().resolve())
    # 只暴露目录、不暴露 dotfile
    for entry in body["entries"]:
        assert entry["isDirectory"] is True
        assert not entry["name"].startswith(".")


async def test_listdir_returns_sorted_subdirs_only(client: AsyncClient, safe_dir):
    for name in ("zeta", "alpha", "middle"):
        os.makedirs(os.path.join(safe_dir, name))
    Path(safe_dir, "a-file.txt").write_text("x")
    os.makedirs(os.path.join(safe_dir, ".hidden"))

    body = (await client.get("/api/fs/listdir", params={"path": safe_dir})).json()
    assert [e["name"] for e in body["entries"]] == ["alpha", "middle", "zeta"]
    assert body["path"] == os.path.realpath(safe_dir)
    assert body["parent"] == os.path.dirname(os.path.realpath(safe_dir))


async def test_listdir_drives_sentinel(client: AsyncClient):
    body = (await client.get("/api/fs/listdir", params={"path": "__drives__"})).json()
    assert body["path"] == "__drives__"
    assert body["parent"] is None
    # POSIX 上虚拟根就是 /
    assert [e["name"] for e in body["entries"]] == ["/"]


async def test_listdir_rejects_relative_path(client: AsyncClient):
    res = await client.get("/api/fs/listdir", params={"path": "relative/dir"})
    assert res.status_code == 400


async def test_listdir_rejects_sensitive_dir(client: AsyncClient):
    res = await client.get("/api/fs/listdir", params={"path": "/etc"})
    assert res.status_code == 403


async def test_listdir_missing_path_is_404(client: AsyncClient, safe_dir):
    res = await client.get("/api/fs/listdir", params={"path": os.path.join(safe_dir, "nope")})
    assert res.status_code == 404


async def test_listdir_file_is_400(client: AsyncClient, safe_dir):
    target = os.path.join(safe_dir, "file.txt")
    Path(target).write_text("x")
    res = await client.get("/api/fs/listdir", params={"path": target})
    assert res.status_code == 400
    assert "Not a directory" in res.json()["error"]


async def test_listdir_root_has_null_parent(client: AsyncClient):
    body = (await client.get("/api/fs/listdir", params={"path": "/"})).json()
    assert body["path"] == "/"
    assert body["parent"] is None
