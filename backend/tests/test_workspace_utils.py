"""is_path_safe —— 路径安检的放行/拒绝边界。

完整的沙箱能力属于 P4；这里只覆盖 P1
用到的 is_path_safe（建会话的 boundPath 校验、/api/fs/listdir 的导航校验）。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from app.security.workspace_utils import is_path_safe, is_path_within


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
