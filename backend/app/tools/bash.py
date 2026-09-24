"""bash 工具：在 workspace 的 effective cwd 里执行命令。

执行链：命令黑名单（硬拦）→ workspace/cwd 校验 → 高危命令审批（挂起）→ 子进程执行。
子进程用 create_subprocess_exec **禁止 shell=True**；POSIX 下 shell 以
`-l -i -c` 跑（登录交互 shell 才有用户 PATH），`start_new_session=True`
独立进程组 —— 超时 / abort 时 killpg 整组杀，防孤儿进程。

stdio 处理（坑最多的一段）：
- stdout/stderr 各起一个并发 reader（单 reader 交替读会管道死锁）；
- 合并进一个 buffer，超 10000 字符硬截断；
- shell 退出后给 reader 0.5s 宽限（后台子进程可能还握着管道的写端），
  超限再杀一次进程组，标 orphanedStdio。

任意退出码都算 ok=True（错误码是模型的观察结果，不是工具故障）；
只有 spawn 失败本身是 ok=False。
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from typing import Any

if os.name != "nt":
    import pwd

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.security.bash_approval import classify_bash_approval
from app.security.shell_bans import find_banned_pattern
from app.security.workspace_utils import assert_path_within_workspace, get_effective_cwd
from app.schemas.dispatch import RunCommandEvidence
from app.services.dispatch_run_evidence import record_run_command
from app.services.fs_service import get_workspace_for_conversation
from app.services.pending_bash_commands import pending_bash_commands
from app.tools.types import ToolContext, ToolDef, ToolResult

DEFAULT_TIMEOUT_MS = 30_000
MIN_TIMEOUT_MS = 1_000
MAX_TIMEOUT_MS = 15 * 60_000
MAX_OUTPUT_CHARS = 10_000
POSIX_ORPHANED_STDIO_GRACE_S = 0.5
LOGIN_INTERACTIVE_SHELLS = {"bash", "zsh"}

PLATFORM = "windows" if os.name == "nt" else "posix"


class BashArgs(BaseModel):
    # LLM 传的是 JSON Schema 里的 camelCase 键名（timeoutMs），必须收下
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    command: str
    cwd: str | None = None
    timeout_ms: int | None = Field(default=None, alias="timeoutMs", gt=0)


def _clamp_timeout(timeout_ms: int | None) -> int:
    if timeout_ms is None:
        return DEFAULT_TIMEOUT_MS
    return max(MIN_TIMEOUT_MS, min(MAX_TIMEOUT_MS, timeout_ms))


def _build_shell_argv(command: str) -> list[str]:
    if PLATFORM == "windows":
        preamble = (
            "$OutputEncoding = [Console]::OutputEncoding = "
            "[System.Text.UTF8Encoding]::new(); "
        )
        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", preamble + command]
    shell = os.environ.get("SHELL") or pwd.getpwuid(os.getuid()).pw_shell or "/bin/sh"
    if os.path.isabs(shell) and os.path.exists(shell) and os.path.basename(shell) in LOGIN_INTERACTIVE_SHELLS:
        return [shell, "-l", "-i", "-c", command]
    return ["sh", "-c", command]


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        if PLATFORM == "windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


async def _drain_pipe(stream, state: dict) -> None:
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            return
        if state["truncated"]:
            continue
        text = chunk.decode("utf-8", errors="replace")
        combined = state["buffer"] + text
        if len(combined) > MAX_OUTPUT_CHARS:
            state["buffer"] = combined[:MAX_OUTPUT_CHARS]
            state["truncated"] = True
        else:
            state["buffer"] = combined


async def _run_shell_command(
    command: str,
    cwd: str,
    timeout_ms: int,
    abort_signal,
    run_id: str | None = None,
    evidence_kind: str | None = None,
) -> dict:
    timeout_s = timeout_ms / 1000
    state: dict[str, Any] = {"buffer": "", "truncated": False}
    timed_out = False
    aborted = False
    orphaned_stdio = False

    argv = _build_shell_argv(command)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=(PLATFORM != "windows"),
        )
    except OSError as err:
        _record_command_evidence(
            run_id,
            command=command,
            cwd=cwd,
            exit_code=None,
            timed_out=False,
            is_error=True,
            evidence_kind=evidence_kind,
            error=str(err),
        )
        return {"spawnFailed": str(err)}

    def _on_abort() -> None:
        nonlocal aborted
        aborted = True
        _kill_tree(proc)

    if abort_signal is not None:
        if abort_signal.aborted:
            _on_abort()
        else:
            abort_signal.add_listener(_on_abort)

    try:
        try:
            returncode = await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            timed_out = True
            _kill_tree(proc)
            try:
                returncode = await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                returncode = proc.returncode if proc.returncode is not None else -1

        readers = [
            asyncio.create_task(_drain_pipe(proc.stdout, state)),
            asyncio.create_task(_drain_pipe(proc.stderr, state)),
        ]
        _, still_running = await asyncio.wait(readers, timeout=POSIX_ORPHANED_STDIO_GRACE_S)
        if still_running:
            orphaned_stdio = True
            _kill_tree(proc)  # 杀掉继承管道写端的后台子进程，让 reader 收到 EOF
            await asyncio.wait(readers, timeout=5)
    finally:
        if abort_signal is not None:
            abort_signal.remove_listener(_on_abort)

    output = state["buffer"]
    if state["truncated"]:
        output += "\n\n[TRUNCATED at 10000 chars]"
    if timed_out:
        output += f"\n\n[KILLED after {timeout_ms / 1000}s timeout]"
    if aborted:
        output += "\n\n[KILLED after run abort]"
    if returncode is not None and returncode < 0:
        output += f"\n\n[KILLED by signal {-returncode}]"
    if orphaned_stdio:
        output += "\n\n[STOPPED background processes after shell exit to close inherited stdio]"

    _record_command_evidence(
        run_id,
        command=command,
        cwd=cwd,
        exit_code=returncode,
        timed_out=timed_out,
        is_error=False,
        evidence_kind=evidence_kind,
    )
    return {
        "cwd": cwd,
        "command": command,
        "exitCode": returncode,
        "output": output,
        "truncated": state["truncated"],
        "timedOut": timed_out,
    }


def _record_command_evidence(
    run_id: str | None,
    *,
    command: str,
    cwd: str,
    exit_code: int | None,
    timed_out: bool,
    is_error: bool,
    evidence_kind: str | None,
    error: str | None = None,
) -> None:
    """执行过的命令按 run 记证据；prepare 标记区分准备命令与验证命令。"""
    if run_id is None:
        return
    record_run_command(
        run_id,
        RunCommandEvidence(
            command=command,
            cwd=cwd,
            exitCode=exit_code,
            timedOut=timed_out,
            isError=is_error,
            prepare=True if evidence_kind == "prepare" else None,
            error=error,
        ),
    )


async def _handle(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        parsed = BashArgs.model_validate(args or {})
    except ValidationError as err:
        return ToolResult(ok=False, error=f"Invalid args: {err}")
    return await execute_bash_command(
        parsed.command, parsed.cwd, parsed.timeout_ms, ctx
    )


async def execute_bash_command(
    command: str,
    cwd: str | None,
    timeout_ms: int | None,
    ctx: ToolContext,
    *,
    evidence_kind: str | None = None,
) -> ToolResult:
    """执行链与 bash 工具入口一致：黑名单 → workspace/cwd → 审批 → 子进程。

    调度器补跑 requiredCommands 也走这里（evidence_kind='prepare'/'verification'），
    保证验证命令和工具命令过的是同一套安全检查。
    """
    banned = find_banned_pattern(command, PLATFORM)
    if banned is not None:
        return ToolResult(ok=False, error=f"Command rejected by safety policy: {banned.pattern}")

    workspace = await get_workspace_for_conversation(ctx.conversation_id)
    if workspace is None:
        return ToolResult(ok=False, error="Workspace not found")

    effective_cwd = get_effective_cwd(workspace)
    if cwd:
        try:
            resolved_cwd = assert_path_within_workspace(workspace, cwd)
        except Exception as err:
            return ToolResult(ok=False, error=str(err))
        if not os.path.isdir(resolved_cwd):
            return ToolResult(ok=False, error=f"cwd is not a directory: {cwd}")
        effective_cwd = resolved_cwd

    approval = classify_bash_approval(command, PLATFORM)
    if approval.required:
        decision = await _wait_bash_approval(ctx, command, effective_cwd, approval.reason)
        if decision is None or not decision.get("approved"):
            return ToolResult(ok=False, error=f"User rejected command execution: {approval.reason}")

    clamped_ms = _clamp_timeout(timeout_ms)
    result = await _run_shell_command(
        command,
        effective_cwd,
        clamped_ms,
        ctx.abort_signal,
        run_id=ctx.run_id,
        evidence_kind=evidence_kind,
    )
    if "spawnFailed" in result:
        return ToolResult(ok=False, error=f"Spawn failed: {result['spawnFailed']}")
    return ToolResult(ok=True, value=result)


async def _wait_bash_approval(ctx: ToolContext, command: str, cwd: str, reason: str) -> dict | None:
    """高危命令挂起等批准；None 表示被 abort。"""
    pending = pending_bash_commands.register(
        conversation_id=ctx.conversation_id,
        agent_id=ctx.agent_id,
        run_id=ctx.run_id,
        command=command,
        cwd=cwd,
        reason=reason,
    )

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    if not pending_bash_commands.attach_resolver(pending["id"], future):
        return None

    def _on_abort() -> None:
        pending_bash_commands.cancel(pending["id"])

    if ctx.abort_signal is not None and ctx.abort_signal.aborted:
        _on_abort()
    elif ctx.abort_signal is not None:
        ctx.abort_signal.add_listener(_on_abort)
    try:
        return await future
    finally:
        if ctx.abort_signal is not None:
            ctx.abort_signal.remove_listener(_on_abort)


BASH_TOOL = ToolDef(
    name="bash",
    description=(
        "Run a shell command in the current workspace directory. "
        "Working directory is always the workspace root (or a subdirectory via cwd). "
        "Each invocation is independent: no interactive stdin, no persistent background servers. "
        "Timeout defaults to 30s (max 15min). "
        "Commands may be subject to user approval for safety."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run"},
            "cwd": {"type": "string", "description": "Working directory inside the workspace (optional)"},
            "timeoutMs": {"type": "integer", "description": "Timeout in milliseconds, 1000-900000 (optional)"},
        },
        "required": ["command"],
    },
    handler=_handle,
)
