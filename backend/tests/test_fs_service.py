"""fs_service 的行为边界：listdir 排序 / dotfile 隐藏 / 逃逸抛错 / 限额文案 / 配额。"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from app.security.workspace_utils import PathOutsideWorkspaceError
from app.services.fs_service import (
    MAX_READ_CHARS,
    MAX_WRITE_BYTES,
    read_file_in_workspace,
    read_if_exists,
    write_file_in_workspace,
    list_dir_in_workspace,
)


@dataclass
class _Ws:
    mode: str
    root_path: str
    bound_path: str | None = None


@pytest.fixture()
def ws_root(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    return str(root)


@pytest.fixture()
def ws(ws_root):
    return _Ws(mode="sandbox", root_path=ws_root)


# ---- list_dir_in_workspace 核心用例 ----


def test_lists_visible_entries_with_directories_first(ws, ws_root):
    os.makedirs(os.path.join(ws_root, "src"))
    with open(os.path.join(ws_root, "README.md"), "w") as fh:
        fh.write("0123456789")  # 10 字节
    with open(os.path.join(ws_root, ".env"), "w") as fh:
        fh.write("secret")

    result = list_dir_in_workspace(ws, "")
    assert result == {
        "relPath": "",
        "absolutePath": ws_root,
        "parent": None,
        "entries": [
            {"name": "src", "isDirectory": True},
            {"name": "README.md", "isDirectory": False, "size": 10},
        ],
    }


def test_listdir_rejects_directory_escapes(ws):
    with pytest.raises(PathOutsideWorkspaceError, match=r'Path "\.\." is outside workspace'):
        list_dir_in_workspace(ws, "..")


def test_listdir_subdirectory_parent(ws, ws_root):
    os.makedirs(os.path.join(ws_root, "src", "nested"))
    result = list_dir_in_workspace(ws, "src/nested")
    assert result["relPath"] == "src/nested"
    assert result["parent"] == "src"
    assert result["entries"] == []


# ---- read ----


def test_read_returns_size_and_content(ws, ws_root):
    with open(os.path.join(ws_root, "notes.md"), "w") as fh:
        fh.write("hello")
    result = read_file_in_workspace(ws, "notes.md")
    assert result["content"] == "hello"
    assert result["size"] == 5
    assert result["truncated"] is False
    assert result["cwd"] == ws_root
    assert result["path"] == "notes.md"


def test_read_truncates_at_50000_chars(ws, ws_root):
    with open(os.path.join(ws_root, "big.txt"), "w") as fh:
        fh.write("x" * (MAX_READ_CHARS + 100))
    result = read_file_in_workspace(ws, "big.txt")
    assert result["truncated"] is True
    assert result["content"].startswith("x" * MAX_READ_CHARS)
    assert result["content"].endswith("\n\n[TRUNCATED at 50000 chars]")


def test_read_rejects_non_file(ws, ws_root):
    os.makedirs(os.path.join(ws_root, "adir"))
    with pytest.raises(ValueError, match="Not a file: adir"):
        read_file_in_workspace(ws, "adir")


def test_read_rejects_too_large_file(ws, ws_root):
    with open(os.path.join(ws_root, "huge.bin"), "wb") as fh:
        fh.write(b"\0" * (1_048_577))
    with pytest.raises(ValueError, match=r"File too large \(.*> 1 MB limit\)"):
        read_file_in_workspace(ws, "huge.bin")


def test_read_if_exists_variants(ws, ws_root):
    with open(os.path.join(ws_root, "exists.txt"), "w") as fh:
        fh.write("data")
    assert read_if_exists(ws, "exists.txt") == "data"
    assert read_if_exists(ws, "missing.txt") is None
    with open(os.path.join(ws_root, "big.txt"), "wb") as fh:
        fh.write(b"\0" * (1_048_577))
    assert read_if_exists(ws, "big.txt") is None


# ---- write ----


def test_write_creates_parent_dirs(ws, ws_root):
    result = write_file_in_workspace(ws, "src/deep/file.txt", "content")
    assert result["bytes"] == len(b"content")
    with open(os.path.join(ws_root, "src", "deep", "file.txt")) as fh:
        assert fh.read() == "content"


def test_write_rejects_oversize_content(ws):
    with pytest.raises(ValueError, match=r"Content too large \(.*> 100 KB limit\)"):
        write_file_in_workspace(ws, "big.txt", "x" * (MAX_WRITE_BYTES + 1))


def test_write_rejects_escapes(ws):
    with pytest.raises(PathOutsideWorkspaceError):
        write_file_in_workspace(ws, "../evil.txt", "x")


def test_write_sandbox_quota_enforced(ws, ws_root):
    # 文件数上限：塞满 1000 个文件后，新文件被拒（覆盖已有文件仍允许）
    target_dir = os.path.join(ws_root, "fill")
    os.makedirs(target_dir)
    for i in range(1000):
        with open(os.path.join(target_dir, f"f{i}.txt"), "w") as fh:
            fh.write("")
    with pytest.raises(ValueError, match=r"Workspace file count exceeded \(1000 \+ 1 > 1000 cap\)"):
        write_file_in_workspace(ws, "fill/one-more.txt", "x")
    # 覆盖既有文件不增加文件数
    result = write_file_in_workspace(ws, "fill/f0.txt", "overwrite")
    assert result["bytes"] == 9


def test_write_local_mode_skips_quota(tmp_path):
    # local 模式没有 1000 文件配额
    local_root = tmp_path / "local-ws"
    local_root.mkdir()
    for i in range(1100):
        with open(local_root / f"f{i}.txt", "w") as fh:
            fh.write("")
    local_ws = _Ws(mode="local", root_path=str(local_root), bound_path=str(local_root))
    result = write_file_in_workspace(local_ws, "one-more.txt", "x")
    assert result["bytes"] == 1
