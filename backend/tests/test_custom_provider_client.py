"""resolve_custom_provider_client_config：provider → key / base_url 解析。

环境变量兜底的行为在测试里显式管理（monkeypatch），避免本机环境泄漏进用例。
"""

from __future__ import annotations

import pytest

from app.adapters.custom_provider_client import resolve_custom_provider_client_config
from app.errors import ServiceError
from app.services.openai_compatible import (
    OPENAI_COMPATIBLE_API_KEY_REQUIRED_ERROR,
    OPENAI_COMPATIBLE_BASE_URL_REQUIRED_ERROR,
)

ENV_KEYS = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ARK_API_KEY")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


async def test_openai_compatible_uses_per_agent_key_and_base_url():
    config = resolve_custom_provider_client_config(
        "openai-compatible",
        "  provider-key  ",
        "  https://dashscope.aliyuncs.com/compatible-mode/v1  ",
    )
    assert config.api_key == "provider-key"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


async def test_openai_compatible_requires_base_url():
    with pytest.raises(ServiceError, match=OPENAI_COMPATIBLE_BASE_URL_REQUIRED_ERROR):
        resolve_custom_provider_client_config("openai-compatible", "provider-key", None)


async def test_openai_compatible_requires_per_agent_key():
    with pytest.raises(ServiceError, match=OPENAI_COMPATIBLE_API_KEY_REQUIRED_ERROR):
        resolve_custom_provider_client_config(
            "openai-compatible", None, "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )


async def test_named_provider_env_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    config = resolve_custom_provider_client_config("openai", None, None)
    assert config.api_key == "env-openai-key"
    assert config.base_url is None  # 走 SDK 默认 endpoint


async def test_deepseek_default_base_url_and_key_error():
    with pytest.raises(ServiceError, match="DEEPSEEK_API_KEY"):
        resolve_custom_provider_client_config("deepseek", None, None)

    config = resolve_custom_provider_client_config("deepseek", "agent-key", None)
    assert config.api_key == "agent-key"
    assert config.base_url == "https://api.deepseek.com/v1"


async def test_volcano_ark_default_base_url_and_key_error():
    with pytest.raises(ServiceError, match="ARK_API_KEY"):
        resolve_custom_provider_client_config("volcano-ark", None, None)

    config = resolve_custom_provider_client_config("volcano-ark", "agent-key", None)
    assert config.api_key == "agent-key"
    assert config.base_url == "https://ark.cn-beijing.volces.com/api/v3"


async def test_per_agent_key_beats_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    config = resolve_custom_provider_client_config("deepseek", "agent-key", None)
    assert config.api_key == "agent-key"


async def test_unsupported_provider():
    with pytest.raises(ServiceError, match="does not support provider"):
        resolve_custom_provider_client_config("anthropic", "key", None)
