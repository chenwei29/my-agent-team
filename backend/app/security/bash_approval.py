"""bash 命令审批分类：哪些命令要挂起等用户批准。

这是一层「增量」防护：黑名单（shell_bans）硬拦破坏性命令，这里只把
「有副作用但不致命」的高危命令（装包、改 git 历史、递归删、docker…）
引向人工审批；不匹配的命令照常直跑。

注意这是行为契约：命中哪条规则就展示哪条 reason（前端面板会原文展示），
调整规则文案要连同前端预期一起考虑。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class BashApproval:
    required: bool
    reason: str


# 每组 (正则, reason)，全部大小写不敏感；首条命中即返回
_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(?:npm|pnpm|yarn|bun)\s+(?:install|i|ci|add|remove|rm|uninstall|update|upgrade)\b", re.I),
        "package manager changes dependencies or downloads packages",
    ),
    (
        re.compile(r"\b(?:npx|bunx)\b|\b(?:pnpm|yarn)\s+dlx\b", re.I),
        "package runner may download and execute packages",
    ),
    (
        re.compile(r"\b(?:pip|pip3|uv)\s+(?:install|add|remove|sync)\b|\bpython3?\s+-m\s+pip\s+install\b", re.I),
        "Python package command may download or change dependencies",
    ),
    (
        re.compile(r"\bgit\s+(?:reset|clean)\b", re.I),
        "git command may discard local changes",
    ),
    (
        re.compile(r"\bgit\s+(?:checkout|restore)\b[\s\S]*\s\.(?:\s|$)", re.I),
        "git command may overwrite workspace files",
    ),
    (
        re.compile(r"\brm\s+-(?:[A-Za-z]*r[A-Za-z]*f|[A-Za-z]*f[A-Za-z]*r)[A-Za-z]*\b", re.I),
        "recursive force delete command",
    ),
    (
        re.compile(r"\bfind\b[\s\S]*\s-delete\b", re.I),
        "find -delete may remove many files",
    ),
    (
        re.compile(r"\b(?:chmod|chown)\b", re.I),
        "permission or ownership change",
    ),
    (
        re.compile(r"\bdocker\s+(?:run|compose|build|push|pull|system|volume|network)\b", re.I),
        "Docker command may affect local containers, images, or network",
    ),
]

_WINDOWS_EXTRA_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bRemove-Item\b[\s\S]*-(?:Recurse|Force)\b", re.I),
        "PowerShell recursive or forced removal",
    ),
    (
        re.compile(r"\b(?:npm|pnpm|yarn|bun)\.cmd\s+(?:install|i|ci|add|remove|rm|uninstall|update|upgrade)\b", re.I),
        "package manager changes dependencies or downloads packages",
    ),
]


def classify_bash_approval(command: str, platform: str) -> BashApproval:
    """非破坏性但高危的命令 → required=True，须等用户批准。"""
    rules = _RULES + (_WINDOWS_EXTRA_RULES if platform == "windows" else [])
    for pattern, reason in rules:
        if pattern.search(command):
            return BashApproval(required=True, reason=reason)
    return BashApproval(required=False, reason="")
