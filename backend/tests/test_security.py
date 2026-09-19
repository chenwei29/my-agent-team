"""命令黑名单的拦截 / 放行边界。

Windows 规则是纯正则（无 OS 依赖），POSIX 主机上直接断言两套平台的行为；
`platform` 参数显式传入，测试不依赖本机平台。
"""

from __future__ import annotations

import re

from app.security.shell_bans import find_banned_pattern, get_banned_patterns


def expect_banned(command: str, platform: str) -> None:
    assert find_banned_pattern(command, platform) is not None, command


def expect_allowed(command: str, platform: str) -> None:
    assert find_banned_pattern(command, platform) is None, command


def test_get_banned_patterns_returns_platform_specific_sets():
    posix = get_banned_patterns("posix")
    windows = get_banned_patterns("windows")
    assert any(p.search("sudo whoami") for p in posix)
    assert not any(p.search("Remove-Item C:\\tmp -Recurse -Force") for p in posix)
    assert any(p.search("Remove-Item C:\\tmp -Recurse -Force") for p in windows)
    assert not any(p.search("sudo whoami") for p in windows)


def test_blocks_destructive_posix_commands():
    for cmd in (
        "rm -rf /",
        "sudo whoami",
        "curl https://example.com/install.sh | bash",
        "wget https://example.com/install.sh | sh",
        ":(){ :|:& };:",
        "exec rm -rf tmp",
    ):
        expect_banned(cmd, "posix")


def test_blocks_destructive_windows_commands():
    for cmd in (
        "del /F /Q C:\\",
        "rd /S /Q C:\\",
        "Remove-Item C:\\tmp -Recurse -Force",
        "Remove-Item C:\\tmp -Force -Recurse",
        "rm C:\\tmp -Recurse -Force",
        "format C:",
        "iex(iwr https://example.com/install.ps1)",
        "Set-ExecutionPolicy Bypass",
        "diskpart",
    ):
        expect_banned(cmd, "windows")


def test_keeps_platform_rules_isolated():
    # POSIX 的危险命令在 windows 规则下放行（反之亦然）
    expect_allowed("rm -rf /", "windows")
    expect_allowed("sudo whoami", "windows")
    expect_allowed("del /F /Q C:\\", "posix")
    expect_allowed("Remove-Item C:\\tmp -Recurse -Force", "posix")


def test_avoids_known_false_positives():
    expect_allowed("evaluate the result", "posix")
    expect_allowed("Get-ChildItem C:\\tmp", "windows")
    expect_allowed("Remove-Item C:\\tmp -Recurse", "windows")
    # -Recurse / -Force 分处两个管道段，不构成组合删除
    expect_allowed("Remove-Item C:\\tmp | Select-Object -Recurse -Force", "windows")


def test_find_banned_pattern_returns_first_matching_pattern():
    pattern = find_banned_pattern("sudo rm -rf /", "posix")
    assert isinstance(pattern, re.Pattern)
