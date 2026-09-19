"""模型上下文窗口表与 token 估算。"""

from __future__ import annotations

from app.utils.model_registry import estimate_tokens, get_model_limits


async def test_known_model_wins_over_provider():
    limits = get_model_limits("deepseek", "deepseek-chat")
    assert limits.context_window == 64_000
    assert limits.output_reserve == 4096  # 默认输出预留

    # reasoning 模型预留更大
    limits = get_model_limits("deepseek", "deepseek-reasoner")
    assert limits.context_window == 128_000
    assert limits.output_reserve == 16_384


async def test_unknown_model_falls_back_to_provider():
    limits = get_model_limits("openai", "totally-new-model")
    assert limits.context_window == 128_000

    limits = get_model_limits("volcano-ark", None)
    assert limits.context_window == 32_000


async def test_no_provider_uses_final_fallback():
    limits = get_model_limits(None, None)
    assert limits.context_window == 200_000


async def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2  # ceil
