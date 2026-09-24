"""全局 API key / endpoint 设置（单行表，PK 固定 'singleton'）。

优先级：agents.api_key > app_settings > process.env，
per-agent override 由 adapter 自行处理（P3）。
"""

from __future__ import annotations

import os

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AppSettings
from app.utils.time import now_ms

SINGLETON_ID = "singleton"

_STRING_FIELDS = (
    "anthropic_api_key",
    "anthropic_base_url",
    "openai_api_key",
    "deepseek_api_key",
    "ark_api_key",
    "deployment_publish_dir",
    "deployment_public_base_url",
)


def _empty() -> dict:
    return {
        "id": SINGLETON_ID,
        "anthropic_api_key": None,
        "anthropic_base_url": None,
        "openai_api_key": None,
        "deepseek_api_key": None,
        "ark_api_key": None,
        "deployment_publish_enabled": False,
        "deployment_publish_dir": None,
        "deployment_public_base_url": None,
        "updated_at": 0,
    }


def _normalize(value: object) -> object:
    """空串归一为 null，避免 "" 与 null 混杂。trim 用户输入。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    trimmed = str(value).strip()
    return trimmed or None


async def get_app_settings(session: AsyncSession) -> dict:
    row = await session.scalar(select(AppSettings).where(AppSettings.id == SINGLETON_ID))
    if row is None:
        return _empty()
    return {c.name: getattr(row, c.name) for c in AppSettings.__table__.columns}


async def update_app_settings(session: AsyncSession, patch: dict, provided: set[str]) -> dict:
    """UPSERT：只在 provided 里出现的字段才动；传 None 清空。"""
    current = await get_app_settings(session)
    next_values = dict(current)

    for key in _STRING_FIELDS:
        if key in provided:
            next_values[key] = _normalize(patch.get(key))
    if "deployment_publish_enabled" in provided:
        next_values["deployment_publish_enabled"] = bool(patch.get("deployment_publish_enabled"))

    next_values["id"] = SINGLETON_ID
    next_values["updated_at"] = now_ms()

    stmt = sqlite_insert(AppSettings).values(**next_values)
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[AppSettings.id],
            set_={
                key: next_values[key]
                for key in next_values
                if key != "id"
            },
        )
    )
    await session.commit()
    return next_values


async def get_effective_api_key(session: AsyncSession, provider: str) -> str | None:
    """app_settings → env var → None。"""
    settings = await get_app_settings(session)
    env_by_provider = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "ark": "ARK_API_KEY",
    }
    field_by_provider = {
        "anthropic": "anthropic_api_key",
        "openai": "openai_api_key",
        "deepseek": "deepseek_api_key",
        "ark": "ark_api_key",
    }
    stored = settings.get(field_by_provider[provider])
    return stored or os.environ.get(env_by_provider[provider]) or None


async def get_effective_anthropic_base_url(session: AsyncSession) -> str | None:
    """app_settings → env var → None（None 表示用官方默认端点）。"""
    settings = await get_app_settings(session)
    return settings.get("anthropic_base_url") or os.environ.get("ANTHROPIC_BASE_URL") or None
