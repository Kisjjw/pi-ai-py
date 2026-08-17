"""用打包进来的真实模型目录验证类型定义。

这是对 types.py 最强的一次检查：39 个厂商、上千个模型，任何字段缺失、
类型不符、compat 归属错误都会在这里暴露。
"""

import pytest

from pi_ai.catalog import DATA_DIR, load_provider_models

PROVIDER_IDS = sorted(path.stem for path in DATA_DIR.glob("*.json") if not path.name.startswith("."))


def test_data_directory_is_populated():
    assert len(PROVIDER_IDS) >= 35, f"expected the synced catalog, got {PROVIDER_IDS}"


# Azure 的端点在运行时由 AZURE_OPENAI_BASE_URL 或 AZURE_OPENAI_RESOURCE_NAME 拼出，
# 目录里的 baseUrl 是空字符串。
RUNTIME_BASE_URL_PROVIDERS = {"azure-openai-responses"}


@pytest.mark.parametrize("provider_id", PROVIDER_IDS)
def test_every_provider_catalog_parses(provider_id):
    models = load_provider_models(provider_id)
    assert models, f"{provider_id} has no models"

    for model_id, model in models.items():
        assert model.id == model_id
        assert model.provider == provider_id
        assert model.api
        if provider_id not in RUNTIME_BASE_URL_PROVIDERS:
            assert model.base_url, f"{provider_id}/{model_id} has no base_url"
        assert model.context_window > 0
        assert model.max_tokens > 0


def test_known_models_carry_expected_metadata():
    anthropic = load_provider_models("anthropic")
    sonnet = next(m for mid, m in anthropic.items() if "sonnet" in mid)
    assert sonnet.api == "anthropic-messages"
    assert sonnet.reasoning is True
    assert "image" in sonnet.input
    assert sonnet.cost.input > 0


def test_compat_is_parsed_into_the_right_type():
    from pi_ai.types import AnthropicMessagesCompat, OpenAICompletionsCompat

    deepseek = load_provider_models("deepseek")
    model = next(iter(deepseek.values()))
    assert isinstance(model.compat, OpenAICompletionsCompat)
    assert model.compat.thinking_format == "deepseek"

    anthropic = load_provider_models("anthropic")
    with_compat = [m for m in anthropic.values() if m.compat is not None]
    for model in with_compat:
        assert isinstance(model.compat, AnthropicMessagesCompat)


# OpenRouter 的自动路由模型把定价填成 -1000000 作为"价格由实际路由到的模型决定"的哨兵。
# TS 的 calculateCost 同样不做钳制，会算出负费用；这里保持一致，不额外处理。
UNPRICED_MODELS = {("openrouter", "openrouter/auto"), ("openrouter", "openrouter/auto-beta")}


def test_cost_calculation_runs_on_every_real_model():
    from pi_ai.cost import calculate_cost
    from pi_ai.types import Usage

    for provider_id in PROVIDER_IDS:
        for model in load_provider_models(provider_id).values():
            usage = Usage(input=1000, output=500, cache_read=100, cache_write=50)
            cost = calculate_cost(model, usage)
            if (provider_id, model.id) in UNPRICED_MODELS:
                assert cost.total < 0
            else:
                assert cost.total >= 0


def test_unpriced_sentinel_models_are_still_present():
    openrouter = load_provider_models("openrouter")
    for _, model_id in UNPRICED_MODELS:
        assert model_id in openrouter
        assert openrouter[model_id].cost.input < 0


def test_thinking_levels_resolve_on_every_real_model():
    from pi_ai.cost import clamp_thinking_level, get_supported_thinking_levels

    for provider_id in PROVIDER_IDS:
        for model in load_provider_models(provider_id).values():
            levels = get_supported_thinking_levels(model)
            assert levels, f"{provider_id}/{model.id} supports no thinking level"
            assert clamp_thinking_level(model, "medium") in levels
            assert clamp_thinking_level(model, "max") in levels


def test_compat_resolves_on_every_real_openai_completions_model():
    """用真实的 21 家厂商配置验证兼容推断不会炸，且产出合法值。"""
    from pi_ai.api.openai_completions_compat import THINKING_FORMATS, get_compat

    checked = 0
    for provider_id in PROVIDER_IDS:
        for model in load_provider_models(provider_id).values():
            if model.api != "openai-completions":
                continue
            compat = get_compat(model)
            assert compat.thinking_format in THINKING_FORMATS, (
                f"{provider_id}/{model.id} -> {compat.thinking_format}"
            )
            assert compat.max_tokens_field in ("max_tokens", "max_completion_tokens")
            assert compat.session_affinity_format in ("openai", "openai-nosession", "openrouter")
            checked += 1

    assert checked > 100, f"expected the real catalog to exercise many models, got {checked}"


def test_known_providers_resolve_to_their_documented_compat():
    from pi_ai.api.openai_completions_compat import get_compat

    deepseek = next(iter(load_provider_models("deepseek").values()))
    assert get_compat(deepseek).thinking_format == "deepseek"
    assert get_compat(deepseek).max_tokens_field == "max_tokens"

    zai = next(iter(load_provider_models("zai").values()))
    assert get_compat(zai).thinking_format == "zai"

    openrouter_models = load_provider_models("openrouter")
    anthropic_via_openrouter = next(m for mid, m in openrouter_models.items() if mid.startswith("anthropic/"))
    compat = get_compat(anthropic_via_openrouter)
    assert compat.cache_control_format == "anthropic"
    assert compat.session_affinity_format == "openrouter"
    assert compat.supports_developer_role is True
