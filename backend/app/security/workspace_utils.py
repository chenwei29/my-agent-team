"""workspace 路径安检 —— 判断目标路径是否允许被读写 / 列目录。

P1 只用得到 is_path_safe（建会话校验 boundPath、listdir API 校验导航目标）。
get_effective_cwd / resolve_safe_path 是 P4 沙箱的活，这里先不引。
"""

from __future__ import annotations

import os
import string
from pathlib import Path

IS_WINDOWS = os.name == "nt"
IS_POSIX = not IS_WINDOWS

# Windows 可用盘符列表，模块级缓存避免重复 stat
_cached_drives: list[str] | None = None


def _available_drives() -> list[str]:
    global _cached_drives
    if _cached_drives is not None:
        return _cached_drives
    if not IS_WINDOWS:
        _cached_drives = []
        return _cached_drives
    drives: list[str] = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            drives.append(root)
    _cached_drives = drives
    return drives


def _system_roots() -> list[str]:
    if not IS_WINDOWS:
        return [
            "/etc",
            "/System",
            "/usr",
            "/bin",
            "/sbin",
            "/var",
            "/private",
            "/Library/Keychains",
        ]
    roots: list[str] = []
    for drive in _available_drives():
        roots.extend(
            [
                os.path.join(drive, "Windows"),
                os.path.join(drive, "Program Files"),
                os.path.join(drive, "Program Files (x86)"),
                os.path.join(drive, "$Recycle.Bin"),
                os.path.join(drive, "System Volume Information"),
                os.path.join(drive, "Recovery"),
            ]
        )
    sys_drive = os.environ.get("SystemDrive", "C:")
    roots.append(os.path.join(f"{sys_drive}\\", "ProgramData"))
    return roots


def _sensitive_segments() -> list[str]:
    shared = [".ssh", ".aws", ".gcloud", ".kube", ".gnupg", ".docker", ".azure"]
    if IS_WINDOWS:
        return [
            *shared,
            "AppData\\Roaming\\Microsoft\\Credentials",
            "AppData\\Local\\Microsoft\\Credentials",
            "AppData\\Roaming\\Microsoft\\Protect",
            "AppData\\Roaming\\gh",
            "AppData\\Roaming\\Claude",
        ]
    return [
        *shared,
        ".config/gh",
        "Library/Keychains",
        "Library/Application Support/Code/User",
    ]


def is_path_within(child: str, parent: str) -> bool:
    """子路径包含判断。Windows 大小写不敏感；POSIX 大小写敏感。"""

    def norm(p: str) -> str:
        # path.resolve 语义：绝对化 + 解析符号链接 + 规范化
        resolved = os.path.realpath(p)
        return resolved.lower() if IS_WINDOWS else resolved

    c = norm(child)
    p = norm(parent)
    return c == p or c.startswith(p + os.sep)


def is_path_safe(abs_path: str) -> bool:
    """拒绝几类明显敏感的目录：

    - 用户的 ssh / aws / gcloud / Windows 凭证等
    - 系统级目录（POSIX: /etc, /System, /usr...；Windows: 每盘符的 \\Windows 等 + \\ProgramData）
    - UNC 设备路径（\\\\?\\ / \\\\.\\）一律拒
    - 用户 home 本身（让用户至少进一层）

    这是「软安全」—— 不阻止恶意路径（用户都能直接编辑 DB 绕过），只是把「随手填错」的坑挡掉。
    """
    # realpath 而非 abspath：必须解析符号链接，macOS 上
    # /tmp → /private/tmp 这类链接必须跟着走到真正的系统根目录，否则安检形同虚设。
    home = os.path.realpath(str(Path.home()))
    normalized = os.path.realpath(abs_path)

    if IS_WINDOWS and (normalized.startswith("\\\\?\\") or normalized.startswith("\\\\.\\")):
        return False
    if IS_WINDOWS and normalized.startswith("\\\\"):
        return False

    home_key = home.lower() if IS_WINDOWS else home
    normalized_key = normalized.lower() if IS_WINDOWS else normalized
    if normalized_key == home_key:
        return False

    for seg in _sensitive_segments():
        if is_path_within(normalized, os.path.join(home, seg)):
            return False

    for root in _system_roots():
        if is_path_within(normalized, root):
            return False

    return True
