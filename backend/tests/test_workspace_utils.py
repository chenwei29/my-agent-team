"""路径安检的放行/拒绝边界 + 工具沙箱（get_effective_cwd / resolve_safe_path）。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.security.workspace_utils import (
    IS_WINDOWS,
    PathOutsideWorkspaceError,
    assert_path_within_workspace,
    get_effective_cwd,
    is_path_safe,
    is_path_within,
    resolve_safe_path,
)


@dataclass
class _WorkspaceRow:
    """workspace 表行的最小替身：沙箱函数只吃 mode/bound_path/root_path 三个字段。"""

    mode: str
    root_path: str
    bound_path: str | None = None


@pytest.fixture()
def ws_root():
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.realpath(os.path.join(tmp, "workspace"))


@pytest.fixture()
def sandbox_ws(ws_root):
    return _WorkspaceRow(mode="sandbox", root_path=ws_root)


def test_home_itself_is_rejected():
    assert is_path_safe(str(Path.home())) is False


def test_sensitive_dirs_under_home_rejected():
    home = Path.home()
    for seg in (".ssh", ".aws", ".gcloud", ".kube", ".gnupg", ".docker", ".azure"):
        assert is_path_safe(str(home / seg)) is False, seg


def test_system_roots_rejected():
    for root in ("/etc", "/usr", "/bin", "/sbin", "/var", "/System"):
        assert is_path_safe(root) is False, root


def test_home_subdir_allowed(safe_dir):
    assert is_path_safe(safe_dir) is True


def test_posix_system_roots_cover_macos_private_symlink():
    """macOS 的 /tmp → /private/tmp，而 /private 是系统根目录，所以必须被拒。

    这条也解释了为什么测试夹具不能直接用 pytest 的 tmp_path。
    """
    assert is_path_safe("/private/tmp") is False
    assert is_path_safe("/tmp/whatever") is False
    assert is_path_safe("/var/folders/xy/T") is False


def test_path_within():
    assert is_path_within("/a/b/c", "/a/b") is True
    assert is_path_within("/a/b", "/a/b") is True
    # 前缀相同但不是子路径
    assert is_path_within("/a/bc", "/a/b") is False
    assert is_path_within("/a", "/a/b") is False


def test_path_within_uses_resolved_paths():
    with tempfile.TemporaryDirectory() as tmp:
        child = os.path.join(tmp, "nested", "file")
        os.makedirs(os.path.dirname(child))
        assert is_path_within(child, tmp) is True


# ---- get_effective_cwd ----


def test_get_effective_cwd_uses_bound_path_for_local_workspaces(ws_root, safe_dir):
    ws = _WorkspaceRow(mode="local", root_path=ws_root, bound_path=safe_dir)
    assert get_effective_cwd(ws) == safe_dir


def test_get_effective_cwd_falls_back_to_root_path(ws_root):
    # sandbox 模式、以及 local 但没绑定 bound_path，都回落 root_path
    assert get_effective_cwd(_WorkspaceRow(mode="sandbox", root_path=ws_root)) == ws_root
    assert get_effective_cwd(_WorkspaceRow(mode="local", root_path=ws_root, bound_path=None)) == ws_root


# ---- is_path_within（沙箱语义） ----


def test_is_path_within_accepts_equal_paths_and_real_descendants(ws_root):
    assert is_path_within(ws_root, ws_root) is True
    assert is_path_within(os.path.join(ws_root, "src", "file.ts"), ws_root) is True


def test_is_path_within_rejects_parent_escapes_and_prefix_traps(ws_root, tmp_path_factory):
    outside = str(tmp_path_factory.mktemp("outside"))
    # 同级目录 + 前缀陷阱：workspace-evil 与 workspace 共享前缀但不是子树
    evil = os.path.join(os.path.dirname(ws_root), "workspace-evil")
    assert is_path_within(outside, ws_root) is False
    assert is_path_within(evil, ws_root) is False


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows 路径大小写不敏感语义")
def test_is_path_within_case_insensitive_on_windows(ws_root):
    assert is_path_within(ws_root.upper(), ws_root.lower()) is True


# ---- resolve_safe_path ----


def test_resolve_safe_path_resolves_relative_and_absolute_paths_inside(ws_root):
    assert resolve_safe_path(_WorkspaceRow(mode="sandbox", root_path=ws_root), "src/file.ts") == os.path.abspath(
        os.path.join(ws_root, "src/file.ts")
    )
    readme = os.path.join(ws_root, "README.md")
    assert resolve_safe_path(_WorkspaceRow(mode="sandbox", root_path=ws_root), readme) == readme


def test_resolve_safe_path_rejects_escapes_and_outside_absolutes(ws_root, tmp_path_factory):
    ws = _WorkspaceRow(mode="sandbox", root_path=ws_root)
    outside = str(tmp_path_factory.mktemp("outside"))
    sibling = os.path.join(os.path.dirname(ws_root), "outside.txt")
    evil = os.path.join(os.path.dirname(ws_root), "workspace-evil")
    assert resolve_safe_path(ws, "..") is None
    assert resolve_safe_path(ws, os.path.join(outside, "x.txt")) is None
    assert resolve_safe_path(ws, sibling) is None
    assert resolve_safe_path(ws, evil) is None


# ---- assert_path_within_workspace ----


def test_assert_path_within_workspace_returns_resolved_paths(ws_root):
    ws = _WorkspaceRow(mode="sandbox", root_path=ws_root)
    assert assert_path_within_workspace(ws, "notes.md") == os.path.abspath(os.path.join(ws_root, "notes.md"))


def test_assert_path_within_workspace_throws_with_context_for_escapes(ws_root):
    ws = _WorkspaceRow(mode="sandbox", root_path=ws_root)
    with pytest.raises(PathOutsideWorkspaceError, match=r'Path "\.\./outside\.txt" is outside workspace'):
        assert_path_within_workspace(ws, "../outside.txt")

