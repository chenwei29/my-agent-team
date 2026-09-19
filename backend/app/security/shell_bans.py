"""bash 命令黑名单 —— 全后端唯一的危险命令数据源。

bash 工具与后续 CLI adapter 的权限回调都从这里取规则：
`find_banned_pattern(command, platform)` 返回首条命中的正则（错误消息里引用
`pattern.pattern` 原文），未命中返回 None。

设计边界（有意为之，别"加强"它）：
- 黑名单只拦"一眼就是破坏性"的命令，真正的兜底是沙箱 + 人工审批；
- POSIX 规则大小写敏感、Windows 规则统一 re.I；
- 正则容忍空格变体（如 ``curl  |  bash``）；
- Windows 规则里的 ``[^|;]*`` 把匹配限制在同一管道段内，
  因此 ``Remove-Item C:\\tmp | Select-Object -Recurse -Force`` 不会被误杀。
"""

from __future__ import annotations

import re

SHARED_BANNED: list[re.Pattern[str]] = []

# POSIX：注意 fork bomb 的正则里 ``:|:`` 需转义管道、``&`` 需转义
POSIX_BANNED: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-rf\s+/"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchmod\s+\d{3,4}\s+/"),
    re.compile(r":\(\)\{\s*:\|:&\s*\}"),
    re.compile(r"curl\s+[^|]*\|\s*(bash|sh)"),
    re.compile(r"wget\s+[^|]*\|\s*(bash|sh)"),
    re.compile(r"\beval\b"),
    re.compile(r"\bexec\b\s+"),
]

# Windows：全部大小写不敏感；路径里反斜杠已双写转义
WINDOWS_BANNED: list[re.Pattern[str]] = [
    re.compile(r"\b(del|erase)\s+/[fsq\s/]*[a-z]:\\?", re.IGNORECASE),
    re.compile(r"\brd\s+/[sq\s/]*[a-z]:\\?", re.IGNORECASE),
    re.compile(r"\bRemove-Item\b[^|;]*-Recurse[^|;]*-Force", re.IGNORECASE),
    re.compile(r"\bRemove-Item\b[^|;]*-Force[^|;]*-Recurse", re.IGNORECASE),
    re.compile(r"\bri\b[^|;]*-Recurse[^|;]*-Force", re.IGNORECASE),
    re.compile(r"\brm\b[^|;]*-Recurse[^|;]*-Force", re.IGNORECASE),
    re.compile(r"\brm\b[^|;]*-Force[^|;]*-Recurse", re.IGNORECASE),
    re.compile(r"\brmdir\b[^|;]*-Recurse[^|;]*-Force", re.IGNORECASE),
    re.compile(r"\brmdir\b[^|;]*-Force[^|;]*-Recurse", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\brestart-computer\b", re.IGNORECASE),
    re.compile(r"\bstop-computer\b", re.IGNORECASE),
    re.compile(r"\breg\s+delete\b", re.IGNORECASE),
    re.compile(r"\bRemove-ItemProperty\b", re.IGNORECASE),
    re.compile(r"\btaskkill\b[^|;]*/im\s*\*", re.IGNORECASE),
    re.compile(r"\bStop-Process\b[^|;]*-Force[^|;]*\*", re.IGNORECASE),
    re.compile(r"Invoke-Expression\s*\(\s*(Invoke-WebRequest|iwr|curl|wget)", re.IGNORECASE),
    re.compile(r"\biex\b\s*\(\s*(iwr|curl|wget|Invoke-WebRequest)", re.IGNORECASE),
    re.compile(r"Set-ExecutionPolicy\s+(Unrestricted|Bypass)", re.IGNORECASE),
    re.compile(r"\bbcdedit\b", re.IGNORECASE),
    re.compile(r"\bdiskpart\b", re.IGNORECASE),
    re.compile(r"\bcipher\s+/w", re.IGNORECASE),
]


def get_banned_patterns(platform: str) -> list[re.Pattern[str]]:
    """按平台取黑名单：共享规则 + 平台专属规则。"""
    return [*SHARED_BANNED, *(WINDOWS_BANNED if platform == "windows" else POSIX_BANNED)]


def find_banned_pattern(command: str, platform: str) -> re.Pattern[str] | None:
    """返回首条命中的规则；错误消息应引用 ``pattern.pattern``。不做任何规范化。"""
    for pattern in get_banned_patterns(platform):
        if pattern.search(command):
            return pattern
    return None
