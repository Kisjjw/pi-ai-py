import pytest

from pi_ai.api.openai_completions_compat import THINKING_FORMATS, detect_compat, get_compat
from pi_ai.types import Model, ModelCost


def make_model(*, provider="custom", base_url="http://x/v1", model_id="m", compat=None) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider=provider,
        base_url=base_url,
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
        compat=compat,
    )


# --- 端点识别 ---


@pytest.mark.parametrize(
    "provider,base_url",
    [
        ("zai", "http://x/v1"),
        ("zai-coding-cn", "http://x/v1"),
        ("custom", "https://api.z.ai/v1"),
        ("custom", "https://open.bigmodel.cn/api/paas/v4"),
    ],
)
def test_zai_endpoints_are_detected(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.thinking_format == "zai"
    assert compat.supports_reasoning_effort is False
    assert compat.max_tokens_field == "max_tokens"


@pytest.mark.parametrize(
    "provider,base_url",
    [("together", "http://x/v1"), ("custom", "https://api.together.ai/v1"), ("custom", "https://api.together.xyz/v1")],
)
def test_together_endpoints_are_detected(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.thinking_format == "together"
    assert compat.supports_strict_mode is False
    assert compat.supports_long_cache_retention is False


@pytest.mark.parametrize(
    "provider,base_url",
    [("moonshotai", "http://x/v1"), ("moonshotai-cn", "http://x/v1"), ("custom", "https://api.moonshot.cn/v1")],
)
def test_moonshot_endpoints_are_detected(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.supports_reasoning_effort is False
    assert compat.supports_strict_mode is False
    assert compat.max_tokens_field == "max_tokens"


@pytest.mark.parametrize("provider,base_url", [("openrouter", "http://x/v1"), ("custom", "https://openrouter.ai/api/v1")])
def test_openrouter_endpoints_are_detected(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.thinking_format == "openrouter"
    assert compat.session_affinity_format == "openrouter"
    assert compat.supports_developer_role is False


@pytest.mark.parametrize("provider,base_url", [("deepseek", "http://x/v1"), ("custom", "https://api.deepseek.com")])
def test_deepseek_endpoints_are_detected(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.thinking_format == "deepseek"
    assert compat.requires_reasoning_content_on_assistant_messages is True
    assert compat.max_tokens_field == "max_tokens"


@pytest.mark.parametrize("provider,base_url", [("xai", "http://x/v1"), ("custom", "https://api.x.ai/v1")])
def test_grok_endpoints_disable_reasoning_effort(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.supports_reasoning_effort is False
    assert compat.supports_store is False
    # grok 不在 use_max_tokens 集合里
    assert compat.max_tokens_field == "max_completion_tokens"


@pytest.mark.parametrize(
    "provider,base_url", [("nvidia", "http://x/v1"), ("custom", "https://integrate.api.nvidia.com/v1")]
)
def test_nvidia_endpoints_are_detected(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.supports_reasoning_effort is False
    assert compat.supports_strict_mode is False
    assert compat.supports_long_cache_retention is False
    assert compat.max_tokens_field == "max_tokens"


@pytest.mark.parametrize("provider,base_url", [("ant-ling", "http://x/v1"), ("custom", "https://api.ant-ling.com/v1")])
def test_ant_ling_endpoints_are_detected(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.thinking_format == "ant-ling"
    assert compat.supports_reasoning_effort is False
    assert compat.supports_long_cache_retention is False


@pytest.mark.parametrize(
    "provider,base_url",
    [("cloudflare-workers-ai", "http://x/v1"), ("custom", "https://api.cloudflare.com/client/v4")],
)
def test_cloudflare_workers_ai_is_detected(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.supports_store is False
    assert compat.supports_long_cache_retention is False
    # workers-ai 不在 use_max_tokens 集合里
    assert compat.max_tokens_field == "max_completion_tokens"


@pytest.mark.parametrize(
    "provider,base_url",
    [("cloudflare-ai-gateway", "http://x/v1"), ("custom", "https://gateway.ai.cloudflare.com/v1")],
)
def test_cloudflare_ai_gateway_is_detected(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.supports_reasoning_effort is False
    assert compat.supports_strict_mode is False
    assert compat.max_tokens_field == "max_tokens"


@pytest.mark.parametrize("provider,base_url", [("cerebras", "http://x/v1"), ("custom", "https://api.cerebras.ai/v1")])
def test_cerebras_is_non_standard_but_keeps_reasoning_effort(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.supports_store is False
    assert compat.supports_reasoning_effort is True


@pytest.mark.parametrize("provider,base_url", [("opencode", "http://x/v1"), ("custom", "https://opencode.ai/v1")])
def test_opencode_is_non_standard(provider, base_url):
    compat = detect_compat(make_model(provider=provider, base_url=base_url))
    assert compat.supports_store is False
    assert compat.supports_developer_role is False


def test_chutes_uses_max_tokens():
    compat = detect_compat(make_model(base_url="https://llm.chutes.ai/v1"))
    assert compat.max_tokens_field == "max_tokens"
    assert compat.supports_store is False


# --- 推断出的默认值 ---


def test_plain_openai_endpoint_gets_standard_defaults():
    compat = detect_compat(make_model(provider="openai", base_url="https://api.openai.com/v1"))
    assert compat.supports_store is True
    assert compat.supports_developer_role is True
    assert compat.supports_reasoning_effort is True
    assert compat.supports_usage_in_streaming is True
    assert compat.supports_finish_reason is True
    assert compat.max_tokens_field == "max_completion_tokens"
    assert compat.requires_tool_result_name is False
    assert compat.requires_assistant_after_tool_result is False
    assert compat.requires_thinking_as_text is False
    assert compat.requires_reasoning_content_on_assistant_messages is False
    assert compat.thinking_format == "openai"
    assert compat.zai_tool_stream is False
    assert compat.supports_thinking_token_budget is False
    assert compat.supports_strict_mode is True
    assert compat.supports_openai_grammar_tools is False
    assert compat.cache_control_format is None
    assert compat.send_session_affinity_headers is False
    assert compat.deferred_tools_mode is None
    assert compat.session_affinity_format == "openai"
    assert compat.supports_long_cache_retention is True
    assert compat.open_router_routing == {}
    assert compat.vercel_gateway_routing == {}
    assert compat.chat_template_kwargs == {}
    assert compat.chat_template_args == {}


def test_openrouter_anthropic_and_openai_models_keep_the_developer_role():
    for model_id in ("anthropic/claude-sonnet-4-5", "openai/gpt-5"):
        compat = detect_compat(make_model(provider="openrouter", model_id=model_id))
        assert compat.supports_developer_role is True

    compat = detect_compat(make_model(provider="openrouter", model_id="google/gemini-2.5-flash"))
    assert compat.supports_developer_role is False


def test_cache_control_format_requires_the_literal_openrouter_provider():
    assert detect_compat(make_model(provider="openrouter", model_id="anthropic/x")).cache_control_format == "anthropic"
    # baseUrl 命中但 provider 不是 openrouter 字面量时不启用
    via_base_url = make_model(
        provider="custom", base_url="https://openrouter.ai/api/v1", model_id="anthropic/x"
    )
    assert detect_compat(via_base_url).cache_control_format is None
    assert detect_compat(make_model(provider="openrouter", model_id="openai/x")).cache_control_format is None


def test_thinking_format_precedence_deepseek_wins_over_openrouter():
    compat = detect_compat(make_model(provider="openrouter", base_url="https://api.deepseek.com"))
    assert compat.thinking_format == "deepseek"


def test_thinking_format_precedence_zai_wins_over_together():
    compat = detect_compat(make_model(provider="zai", base_url="https://api.together.ai/v1"))
    assert compat.thinking_format == "zai"


def test_detected_thinking_format_is_always_a_known_value():
    for provider in ("openai", "zai", "together", "ant-ling", "openrouter", "deepseek", "custom"):
        assert detect_compat(make_model(provider=provider)).thinking_format in THINKING_FORMATS


# --- 显式 compat 覆盖 ---


def test_model_compat_overrides_detected_values():
    model = make_model(
        provider="deepseek",
        compat={"thinkingFormat": "qwen", "supportsStore": True, "maxTokensField": "max_completion_tokens"},
    )
    compat = get_compat(model)
    assert compat.thinking_format == "qwen"
    assert compat.supports_store is True
    assert compat.max_tokens_field == "max_completion_tokens"
    # 未覆盖的字段仍来自推断
    assert compat.requires_reasoning_content_on_assistant_messages is True


def test_get_compat_without_model_compat_equals_detect():
    model = make_model(provider="groq")
    assert get_compat(model) == detect_compat(model)


def test_false_values_in_model_compat_are_respected():
    model = make_model(provider="openai", base_url="https://api.openai.com/v1", compat={"supportsStore": False})
    assert get_compat(model).supports_store is False


def test_open_router_routing_does_not_fall_back_to_detected():
    model = make_model(provider="openrouter")
    assert get_compat(model).open_router_routing == {}

    model = make_model(provider="openrouter", compat={"openRouterRouting": {"order": ["anthropic"]}})
    assert get_compat(model).open_router_routing == {"order": ["anthropic"]}


def test_deferred_tools_mode_and_thinking_budget_pass_through():
    model = make_model(compat={"deferredToolsMode": "kimi", "supportsThinkingTokenBudget": True})
    compat = get_compat(model)
    assert compat.deferred_tools_mode == "kimi"
    assert compat.supports_thinking_token_budget is True


def test_cache_control_format_can_be_set_explicitly():
    model = make_model(compat={"cacheControlFormat": "anthropic"})
    assert get_compat(model).cache_control_format == "anthropic"


def test_compat_assigned_after_construction_is_still_parsed():
    """构造后直接赋值不会触发 pydantic 校验器，get_compat 要自己兜住。

    不兜的话所有开关会静默回退到推断值，错得没有任何提示。
    """
    model = make_model(provider="openai", base_url="https://api.openai.com/v1")
    model.compat = {"supportsFinishReason": False, "thinkingFormat": "qwen"}
    compat = get_compat(model)
    assert compat.supports_finish_reason is False
    assert compat.thinking_format == "qwen"
