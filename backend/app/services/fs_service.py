"""workspace 文件系统核心：读 / 写 / 列目录 + 沙箱配额。

纯同步函数操作 workspace 行（只吃 mode / bound_path / root_path 三个字段，
测试可以直接传替身），DB 查询走短生命周期的 SessionLocal —— 工具 handler 可能
挂起数分钟等审批，绝不能把 session 一起吊着。

限制常量（前后端契约）：
- 单次读 1MB / 50000 字符（超出截断并标记 truncated）
- 单次写 100KB
- sandbox 模式总量配额 100MB / 1000 文件（local 模式不设限）
"""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import select

from app.db.models import Conversation, Workspace
from app.db.session import SessionLocal
from app.security.workspace_utils import (
    PathOutsideWorkspaceError,
    get_effective_cwd,
    resolve_safe_path,
)

MAX_READ_BYTES = 1_048_576
MAX_READ_CHARS = 50_000
MAX_WRITE_BYTES = 100 * 1024
SANDBOX_TOTAL_BYTES = 100 * 1024 * 1024
SANDBOX_TOTAL_FILES = 1000

TRUNCATED_READ_SUFFIX = "\n\n[TRUNCATED at 50000 chars]"


async def get_workspace_for_conversation(conversation_id: str) -> Workspace | None:
    async with SessionLocal() as session:
        return await session.scalar(select(Workspace).where(Workspace.conversation_id == conversation_id))


async def get_conversation_approval_mode(conversation_id: str) -> str:
    """fs_write 的审批模式；会话不存在或字段为空时默认 review。"""
    async with SessionLocal() as session:
        row = await session.scalar(select(Conversation).where(Conversation.id == conversation_id))
    if row is None or not row.fs_write_approval_mode:
        return "review"
    return row.fs_write_approval_mode


def read_file_in_workspace(workspace: Any, target: str) -> dict:
    """读文件。返回 {path, absolutePath, cwd, size, content, truncated}。

    非文件 / 超限 / 路径逃逸均抛异常（文案是给 LLM 和前端看的契约）。
    """
    resolved = resolve_safe_path(workspace, target)
    if resolved is None:
        raise PathOutsideWorkspaceError(f'Path "{target}" is outside workspace')

    if not os.path.isfile(resolved):
        raise ValueError(f"Not a file: {target}")

    size = os.path.getsize(resolved)
    if size > MAX_READ_BYTES:
        raise ValueError(f"File too large ({size / 1048576:.2f} MB > 1 MB limit)")

    with open(resolved, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    truncated = False
    content = raw
    if len(raw) > MAX_READ_CHARS:
        content = raw[:MAX_READ_CHARS] + TRUNCATED_READ_SUFFIX
        truncated = True

    return {
        "path": target,
        "absolutePath": resolved,
        "cwd": get_effective_cwd(workspace),
        "size": size,
        "content": content,
        "truncated": truncated,
    }


def read_if_exists(workspace: Any, target: str) -> str | None:
    """供 fs_write review 模式取 oldContent：缺失 / 非文件 / 超 1MB（diff 不动它）一律 None。"""
    try:
        resolved = resolve_safe_path(workspace, target)
        if resolved is None or not os.path.isfile(resolved):
            return None
        if os.path.getsize(resolved) > MAX_READ_BYTES:
            return None
        with open(resolved, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def write_file_in_workspace(workspace: Any, target: str, content: str) -> dict:
    """写文件。检查顺序（顺序即契约）：100KB 限制 → 路径逃逸 → sandbox 配额 → 落盘。"""
    data = content.encode("utf-8")
    if len(data) > MAX_WRITE_BYTES:
        raise ValueError(f"Content too large ({len(data) / 1024:.1f} KB > 100 KB limit)")

    resolved = resolve_safe_path(workspace, target)
    if resolved is None:
        raise PathOutsideWorkspaceError(f'Path "{target}" is outside workspace')

    if workspace.mode == "sandbox":
        usage = _scan_workspace_usage(workspace.root_path)
        if usage[0] + len(data) > SANDBOX_TOTAL_BYTES:
            raise ValueError(
                f"Workspace quota exceeded ({usage[0] / 1048576:.1f} MB used + "
                f"{len(data) / 1024:.1f} KB > 100 MB cap)"
            )
        if not os.path.exists(resolved) and usage[1] + 1 > SANDBOX_TOTAL_FILES:
            raise ValueError(f"Workspace file count exceeded ({usage[1]} + 1 > 1000 cap)")

    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "wb") as fh:
        fh.write(data)

    return {
        "path": target,
        "absolutePath": resolved,
        "cwd": get_effective_cwd(workspace),
        "bytes": len(data),
    }


def list_dir_in_workspace(workspace: Any, target: str) -> dict:
    """列目录。dotfile 隐藏、目录优先再按名字排。返回 {relPath, absolutePath, parent, entries}。"""
    resolved = resolve_safe_path(workspace, target)
    if resolved is None:
        raise PathOutsideWorkspaceError(f'Path "{target}" is outside workspace')

    if not os.path.isdir(resolved):
        raise ValueError(f"Not a directory: {target or '(root)'}")

    entries: list[dict] = []
    try:
        raw = list(os.scandir(resolved))
    except OSError:
        raw = []

    for entry in raw:
        if entry.name.startswith("."):
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        item: dict[str, Any] = {"name": entry.name, "isDirectory": is_dir}
        if not is_dir:
            try:
                item["size"] = entry.stat().st_size
            except OSError:
                pass
        entries.append(item)

    entries.sort(key=lambda e: (not e["isDirectory"], e["name"]))

    rel_path = target or ""
    parent: str | None = None
    if rel_path:
        # parent 用 posix 风格展示（前端直接拼路径用）
        normalized = "/".join(p for p in rel_path.split("/") if p and p != ".")
        segments = normalized.split("/") if normalized else []
        if len(segments) >= 2:
            parent = "/".join(segments[:-1])
        else:
            parent = ""

    return {
        "relPath": rel_path,
        "absolutePath": resolved,
        "parent": parent,
        "entries": entries,
    }


def _scan_workspace_usage(root_path: str) -> tuple[int, int]:
    """统计 (总字节数, 文件数)。DFS + realpath visited 集合防符号链接环。"""
    total_bytes = 0
    total_files = 0
    visited: set[str] = set()
    stack: list[str] = [root_path]

    while stack:
        current = stack.pop()
        try:
            real = os.path.realpath(current)
        except OSError:
            continue
        if real in visited:
            continue
        visited.add(real)
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file():
                    total_files += 1
                    total_bytes += entry.stat().st_size
            except OSError:
                continue

    return total_bytes, total_files
