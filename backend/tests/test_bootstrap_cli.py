"""`python -m app.db.bootstrap_cli` 的真实入口测试。

为什么单独测：E2E 就靠这个 CLI 建表 + 插 mock agent（跑之前先清空数据目录），
而 `_main()` 里曾经因为参数名和函数同名（`insert_mock_agent`）导致
`--insert-mock-agent` 直接 `'bool' object is not callable` —— 只有走真实命令行才会暴露。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db.bootstrap_cli import E2E_MOCK_AGENT_ID

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _run_cli(data_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "app.db.bootstrap_cli", *args],
        cwd=BACKEND_DIR,
        env={
            **os.environ,
            "AGENTHUB_DATA_DIR": str(data_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )


def _agent_ids(db_path: Path) -> set[str]:
    import sqlite3

    connection = sqlite3.connect(db_path)
    try:
        return {row[0] for row in connection.execute("SELECT id FROM agents")}
    finally:
        connection.close()


def test_cli_creates_db_and_seeds_builtin_agents(tmp_path: Path):
    result = _run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "agenthub.db").exists()
    assert (tmp_path / "workspaces").is_dir()
    assert {"ag_orchestrator", "ag_pm", "ag_designer", "ag_frontend", "ag_reviewer"} <= _agent_ids(
        tmp_path / "agenthub.db"
    )


def test_cli_insert_mock_agent_is_idempotent(tmp_path: Path):
    first = _run_cli(tmp_path, "--insert-mock-agent")
    assert first.returncode == 0, first.stderr
    assert E2E_MOCK_AGENT_ID in _agent_ids(tmp_path / "agenthub.db")

    second = _run_cli(tmp_path, "--insert-mock-agent")
    assert second.returncode == 0, second.stderr
    assert "skip" in second.stdout or "insert" in second.stdout


@pytest.mark.parametrize("flag", ["--insert-mock-agent"])
def test_cli_flag_does_not_shadow_the_function(flag: str, tmp_path: Path):
    """回归护栏：参数名不能和被调用的函数同名。"""
    result = _run_cli(tmp_path, flag)
    assert "not callable" not in (result.stdout + result.stderr)
    assert result.returncode == 0
