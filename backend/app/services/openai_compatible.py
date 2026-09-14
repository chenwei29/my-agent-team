"""OpenAI-compatible provider 的 Base URL / API Key 校验：通过返回 None，否则返回错误文案。

错误文案会被前端原样展示，措辞属于对外契约的一部分，改动前先确认前端展示路径。
"""

from __future__ import annotations

from urllib.parse import urlparse

OPENAI_COMPATIBLE_BASE_URL_REQUIRED_ERROR = (
    "OpenAI-compatible provider 必须填写 Chat Completions Base URL，"
    "例如 https://dashscope.aliyuncs.com/compatible-mode/v1"
)

OPENAI_COMPATIBLE_BASE_URL_FORMAT_ERROR = (
    "OpenAI-compatible Base URL 必须是完整 URL，"
    "例如 https://dashscope.aliyuncs.com/compatible-mode/v1"
)

OPENAI_COMPATIBLE_API_KEY_REQUIRED_ERROR = (
    "OpenAI-compatible provider 必须为该 Agent 单独填写 API Key"
)


def validate_openai_compatible_base_url(
    provider: str | None, base_url: str | None
) -> str | None:
    if provider != "openai-compatible":
        return None

    trimmed = (base_url or "").strip()
    if not trimmed:
        return OPENAI_COMPATIBLE_BASE_URL_REQUIRED_ERROR

    # 严格度按「完整 URL」来卡：必须有 scheme + host，缺一即格式错误
    parsed = urlparse(trimmed)
    if not parsed.scheme or not parsed.netloc:
        return OPENAI_COMPATIBLE_BASE_URL_FORMAT_ERROR

    return None


def validate_openai_compatible_api_key(provider: str | None, api_key: str | None) -> str | None:
    if provider != "openai-compatible":
        return None
    return None if (api_key or "").strip() else OPENAI_COMPATIBLE_API_KEY_REQUIRED_ERROR
