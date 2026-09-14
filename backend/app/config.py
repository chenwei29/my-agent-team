"""数据目录 / DB 路径 / workspace 根 —— 全部由 AGENTHUB_DATA_DIR 派生。

AGENTHUB_DATA_DIR 指定数据根目录（DB 与 workspaces 都在它下面），
未设置时用仓库根的默认目录；E2E 靠它做隔离，避免测试数据落到本地开发目录。
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
REPO_ROOT = BASE_DIR.parent

DEFAULT_DATA_DIR = REPO_ROOT / ".agenthub-data"


def _resolve_data_dir() -> Path:
    """AGENTHUB_DATA_DIR 优先；未设置时固定在仓库根，避免 cwd 不同导致数据分裂。"""
    raw = os.environ.get("AGENTHUB_DATA_DIR")
    if raw and raw.strip():
        return Path(raw.strip()).expanduser().resolve()
    return DEFAULT_DATA_DIR


class Settings:
    def __init__(self) -> None:
        self.data_dir = _resolve_data_dir()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agenthub.db"

    @property
    def workspaces_root(self) -> Path:
        return self.data_dir / "workspaces"

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path}"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_root.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
