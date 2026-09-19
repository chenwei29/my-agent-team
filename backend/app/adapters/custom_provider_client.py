"""Custom agent 的 provider → 客户端配置解析：决定用哪个 API key、哪个 base URL。

被 custom adapter 在每次 run 开始时调用；key 缺失在这里抛错（而不是启动时拒绝服务，
用户可能只配置了部分 provider）。错误文案会被前端原样展示，措辞是对外契约。
"""

from __future__ import annotations

import os

from app.errors import ServiceError
from app.services.openai_compatible import (
    validate_openai_compatible_api_key,
    validate_openai_compatible_base_url,
)

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_VOLCANO_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class CustomProviderClientConfig:
    """传给 OpenAI 客户端的连接参数；base_url 为 None 时走 SDK 默认（api.openai.com）。"""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url


def resolve_custom_provider_client_config(
    provider: str,
    override_key: str | None,
    api_base_url: str | None,
) -> CustomProviderClientConfig:
    """override_key 是 per-agent 的 agents.api_key；env 变量是最后兜底。

    注意 openai-compatible 不做 env 兜底：它的 key 和 endpoint 成对配置，
    全局层面没有「对应的」环境变量可回退。
    """
    if provider == "deepseek":
        api_key = (override_key or "").strip() or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ServiceError("DEEPSEEK_API_KEY not set and agent has no apiKey")
        return CustomProviderClientConfig(api_key, DEFAULT_DEEPSEEK_BASE_URL)

    if provider == "volcano-ark":
        api_key = (override_key or "").strip() or os.environ.get("ARK_API_KEY")
        if not api_key:
            raise ServiceError("ARK_API_KEY not set and agent has no apiKey")
        return CustomProviderClientConfig(api_key, DEFAULT_VOLCANO_ARK_BASE_URL)

    if provider == "openai":
        api_key = (override_key or "").strip() or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ServiceError("OPENAI_API_KEY not set and agent has no apiKey")
        return CustomProviderClientConfig(api_key)

    if provider == "openai-compatible":
        base_url_error = validate_openai_compatible_base_url(provider, api_base_url)
        if base_url_error:
            raise ServiceError(base_url_error)
        key_error = validate_openai_compatible_api_key(provider, override_key)
        if key_error:
            raise ServiceError(key_error)
        return CustomProviderClientConfig((override_key or "").strip(), (api_base_url or "").strip())

    raise ServiceError(f'CustomAgentAdapter does not support provider "{provider}" yet')
