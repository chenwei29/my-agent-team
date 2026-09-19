"""/api/conversations/{id}/fs/* —— FileTab 手动文件端点的行为与错误码映射。"""

from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_fs_roundtrip(client, conversation):
    conv_id = conversation["id"]

    # 空根目录
    res = await client.get(f"/api/conversations/{conv_id}/fs/listdir")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["relPath"] == ""
    assert body["parent"] is None
    assert body["entries"] == []

    # 手动写（不走审批）
    res = await client.post(f"/api/conversations/{conv_id}/fs/write", json={"path": "hello.txt", "content": "hi"})
    assert res.status_code == 200, res.text
    assert res.json()["bytes"] == 2

    # listdir 能看到
    res = await client.get(f"/api/conversations/{conv_id}/fs/listdir")
    entries = res.json()["entries"]
    assert entries == [{"name": "hello.txt", "isDirectory": False, "size": 2}]

    # 读回来
    res = await client.get(f"/api/conversations/{conv_id}/fs/read", params={"path": "hello.txt"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["content"] == "hi"
    assert body["truncated"] is False


@pytest.mark.asyncio
async def test_fs_read_requires_path(client, conversation):
    res = await client.get(f"/api/conversations/{conversation['id']}/fs/read")
    assert res.status_code == 400
    assert res.json()["error"] == "path required"


@pytest.mark.asyncio
async def test_fs_escape_maps_to_403(client, conversation):
    conv_id = conversation["id"]
    res = await client.get(f"/api/conversations/{conv_id}/fs/read", params={"path": "../secrets"})
    assert res.status_code == 403, res.text
    res = await client.post(
        f"/api/conversations/{conv_id}/fs/write", json={"path": "../evil.txt", "content": "x"}
    )
    assert res.status_code == 403, res.text


@pytest.mark.asyncio
async def test_fs_errors(client, conversation):
    conv_id = conversation["id"]

    # 读不存在的文件
    res = await client.get(f"/api/conversations/{conv_id}/fs/read", params={"path": "missing.txt"})
    assert res.status_code == 400
    assert "Not a file" in res.json()["error"]

    # 写超大内容 → 413
    res = await client.post(
        f"/api/conversations/{conv_id}/fs/write", json={"path": "big.txt", "content": "x" * (100 * 1024 + 1)}
    )
    assert res.status_code == 413, res.text
    assert "too large" in res.json()["error"]

    # 不存在的会话 → 404
    res = await client.get("/api/conversations/conv_missing/fs/listdir")
    assert res.status_code == 404
    assert res.json()["error"] == "Workspace not found"
