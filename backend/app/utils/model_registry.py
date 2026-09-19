"""模型上下文窗口 / 输出预留 token 表 + 粗粒度 token 估算。

用于跨 run 历史注入的 token 预算：historyBudget = contextWindow - outputReserve
- estimate(systemPrompt) - estimate(currentUser) - 安全余量。

维护说明：模型常变（厂商上下文窗常扩），表里是相对保守的下限。没在表里的
model id 走 provider fallback，也能跑，只是预算保守一点。
"""

from __future__ import annotations

import math

# 整个会话能装进 LLM 一次调用的总 token 上限（input + output），按 provider 兜底
_PROVIDER_FALLBACK_CONTEXT: dict[str, int] = {
    "anthropic": 200_000,
    "openai": 128_000,
    "deepseek": 64_000,
    "volcano-ark": 32_000,
    "openai-compatible": 128_000,
}

# reasoning 模型实际需要更多（thinking 也吃 token），4K 兜底够用
_DEFAULT_OUTPUT_RESERVE = 4096

_KNOWN_MODELS: dict[str, dict[str, int]] = {
    # DeepSeek
    "deepseek-chat": {"context": 64_000},
    "deepseek-v4-flash": {"context": 64_000},
    "deepseek-v4": {"context": 64_000},
    "deepseek-reasoner": {"context": 128_000, "outputReserve": 16_384},
    "deepseek-r1": {"context": 128_000, "outputReserve": 16_384},
    # OpenAI
    "gpt-4o": {"context": 128_000},
    "gpt-4o-mini": {"context": 128_000},
    "gpt-4-turbo": {"context": 128_000},
    "gpt-4": {"context": 8_192},
    "gpt-3.5-turbo": {"context": 16_385},
    "o1": {"context": 200_000, "outputReserve": 32_768},
    "o1-mini": {"context": 128_000, "outputReserve": 16_384},
    # Anthropic
    "claude-opus-4-7": {"context": 200_000},
    "claude-opus-4-7[1m]": {"context": 1_000_000},
    "claude-sonnet-4-6": {"context": 200_000},
    "claude-opus-4-6": {"context": 200_000},
    "claude-opus-4-5": {"context": 200_000},
    "claude-sonnet-4-5": {"context": 200_000},
    "claude-3-5-sonnet-latest": {"context": 200_000},
    "claude-haiku-4-5-20251001": {"context": 200_000},
    # Volcano Ark / 豆包
    "doubao-seed-2-0-lite-260428": {"context": 32_000},
    "doubao-1-5-pro-256k": {"context": 256_000},
    "doubao-pro-128k": {"context": 128_000},
}


class ModelLimits:
    def __init__(self, context_window: int, output_reserve: int) -> None:
        self.context_window = context_window
        self.output_reserve = output_reserve


def get_model_limits(provider: str | None, model_id: str | None) -> ModelLimits:
    if model_id and model_id in _KNOWN_MODELS:
        entry = _KNOWN_MODELS[model_id]
        return ModelLimits(entry["context"], entry.get("outputReserve", _DEFAULT_OUTPUT_RESERVE))
    if provider and provider in _PROVIDER_FALLBACK_CONTEXT:
        return ModelLimits(_PROVIDER_FALLBACK_CONTEXT[provider], _DEFAULT_OUTPUT_RESERVE)
    # 最终兜底（没有 modelProvider 字段的 adapter 也落在这里）
    return ModelLimits(200_000, _DEFAULT_OUTPUT_RESERVE)


def estimate_tokens(text: str) -> int:
    """粗粒度估算：4 字符 ≈ 1 token。中英混合误差 10-20% 量级，对预算决策够用。"""
    return math.ceil(len(text) / 4)
