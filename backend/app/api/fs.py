"""`/api/fs/listdir` —— 列出指定目录下的**子目录**（DirPickerDialog 用）：
- path 不传：默认 home
- path === '__drives__'：返回可用盘符（Windows 虚拟根；POSIX 上返回 /）
- 其他：必须绝对路径 + 是目录 + 通过 is_path_safe
- 隐藏 dotfile，只暴露目录，按名字排序
- 根目录时 parent 为 null；Windows 盘符根的 parent 为 '__drives__'
"""

from __future__ import annotations

import os
import string
from pathlib import Path

from fastapi import APIRouter, Query

from app.errors import HttpError
from app.schemas.entities import ListDirResponse
from app.security.workspace_utils import IS_WINDOWS, is_path_safe

router = APIRouter(prefix="/api/fs", tags=["fs"])

DRIVES_SENTINEL = "__drives__"

# Windows 已知隐藏 / 系统目录名（大小写不敏感）。文件系统的 stat 拿不到 hidden attribute，
# 只能按名单硬编码，已覆盖 95% 噪音目录。
_WINDOWS_HIDDEN_NAMES = {
    n.lower()
    for n in (
        "AppData",
        "$Recycle.Bin",
        "System Volume Information",
        "Recovery",
        "PerfLogs",
        "Config.Msi",
        "MSOCache",
        "OneDriveTemp",
        "ProgramData",
    )
}


def _list_available_drives() -> list[str]:
    if not IS_WINDOWS:
        return ["/"]
    drives: list[str] = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            drives.append(root)
    return drives


@router.get("/listdir", response_model=ListDirResponse)
async def list_dir(path: str | None = Query(default=None)) -> dict:
    home = str(Path.home().resolve())
    target = (path or "").strip() or home

    if target == DRIVES_SENTINEL:
        drives = _list_available_drives()
        return {
            "path": DRIVES_SENTINEL,
            "parent": None,
            "entries": [
                {"name": d.rstrip("\\/") or d, "isDirectory": True, "path": d} for d in drives
            ],
        }

    if not os.path.isabs(target):
        raise HttpError(400, "path must be absolute")

    resolved = os.path.realpath(target)

    # 允许浏览 home 自身（用作起点）但仍走 is_path_safe 拦截已知敏感子路径
    if resolved != home and not is_path_safe(resolved):
        raise HttpError(403, "Path not allowed")

    if not os.path.exists(resolved):
        raise HttpError(404, "Path does not exist")
    if not os.path.isdir(resolved):
        raise HttpError(400, "Not a directory")

    try:
        raw = list(os.scandir(resolved))
    except OSError as err:
        raise HttpError(403, f"Cannot read directory: {err}") from err

    entries = []
    for entry in raw:
        name = entry.name
        if name.startswith("."):
            continue
        if IS_WINDOWS and name.lower() in _WINDOWS_HIDDEN_NAMES:
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if not is_dir:
            continue
        entries.append({"name": name, "isDirectory": True})
    entries.sort(key=lambda e: e["name"])

    parent_dir = os.path.dirname(resolved)
    if parent_dir != resolved:
        parent: str | None = parent_dir
    else:
        # 已到根。Windows 盘符根暴露虚拟 drives 列表作为上一级
        parent = DRIVES_SENTINEL if IS_WINDOWS else None

    return {"path": resolved, "parent": parent, "entries": entries}
