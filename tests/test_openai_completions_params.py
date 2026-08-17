from dataclasses import replace

from pi_ai.api.openai_completions_compat import detect_compat
from pi_ai.api.openai_completions_params import build_params, resolve_cache_retention
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    Tool,
    ToolCall,
    ToolResultMessage,
    user_text,
)


def make_model(
    *,
    provider="acme",
    base_url="https://acme.test/v1",
    model_id="m",
    reasoning=True,
    level_map=None,
    compat=None,
    max_tokens=4096,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider=provider,
        base_url=base_url,
        reasoning=reasoning,
        thinking_level_map=level_map,
        input=["text"],
        cost=ModelCost(),
        context_window=100000,
        max_tokens=max_tokens,
        compat=compat,
    )


CTX = Context(messages=[user_text("hi")])


def params_for(model, options=None, compat=None, cache_retention="short"):
    return build_params(
        model,
        CTX,
        options or {},
        compat if compat is not None else detect_compat(model),
        cache_retention,
    )


# --- 基础字段 ---


def test_core_fields_are_always_present():
    params = params_for(make_model())
    assert params["model"] == "m"
    assert params["stream"] is True
    assert params["messages"] == [{"role": "user", "content": "hi"}]


def test_usage_in_streaming_is_requested_by_default():
    assert params_for(make_model())["stream_options"] == {"include_usage": True}


def test_usage_in_streaming_can_be_disabled():
    compat = replace(detect_compat(make_model()), supports_usage_in_streaming=False)
    assert "stream_options" not in params_for(make_model(), compat=compat)


def test_store_false_is_sent_when_supported():
    assert params_for(make_model())["store"] is False


def test_store_is_omitted_when_unsupported():
    compat = replace(detect_compat(make_model()), supports_store=False)
    assert "store" not in params_for(make_model(), compat=compat)


def test_max_tokens_field_follows_compat():
    params = params_for(make_model(), {"max_tokens": 500})
    assert params["max_completion_tokens"] == 500

    compat = replace(detect_compat(make_model()), max_tokens_field="max_tokens")
    params = params_for(make_model(), {"max_tokens": 500}, compat=compat)
    assert params["max_tokens"] == 500


def test_temperature_is_only_sent_when_given():
    assert "temperature" not in params_for(make_model())
    assert params_for(make_model(), {"temperature": 0})["temperature"] == 0


# --- 缓存键 ---


def test_openai_endpoint_sends_a_prompt_cache_key():
    model = make_model(provider="openai", base_url="https://api.openai.com/v1")
    params = params_for(model, {"session_id": "s1"})
    assert params["prompt_cache_key"] == "s1"
    assert "prompt_cache_retention" not in params


def test_openai_endpoint_omits_the_cache_key_when_retention_is_none():
    model = make_model(provider="openai", base_url="https://api.openai.com/v1")
    params = params_for(model, {"session_id": "s1"}, cache_retention="none")
    assert "prompt_cache_key" not in params


def test_long_retention_adds_the_24h_field():
    model = make_model(provider="openai", base_url="https://api.openai.com/v1")
    params = params_for(model, {"session_id": "s1"}, cache_retention="long")
    assert params["prompt_cache_retention"] == "24h"


def test_long_retention_is_skipped_when_unsupported():
    compat = replace(detect_compat(make_model()), supports_long_cache_retention=False)
    params = params_for(make_model(), {"session_id": "s1"}, compat=compat, cache_retention="long")
    assert "prompt_cache_retention" not in params
    assert "prompt_cache_key" not in params


def test_non_openai_endpoint_needs_long_retention_for_a_cache_key():
    params = params_for(make_model(), {"session_id": "s1"})
    assert "prompt_cache_key" not in params

    params = params_for(make_model(), {"session_id": "s1"}, cache_retention="long")
    assert params["prompt_cache_key"] == "s1"


# --- 工具 ---


def test_tools_are_converted_when_present():
    ctx = Context(messages=[user_text("hi")], tools=[Tool(name="t", description="d", parameters={})])
    params = build_params(make_model(), ctx, {}, detect_compat(make_model()), "short")
    assert params["tools"][0]["function"]["name"] == "t"


def test_empty_tools_array_is_sent_when_history_has_tool_calls():
    ctx = Context(
        messages=[
            AssistantMessage(
                content=[ToolCall(id="c1", name="t", arguments={})],
                api="openai-completions",
                provider="acme",
                model="m",
                stop_reason="toolUse",
                timestamp=1,
            ),
            ToolResultMessage(tool_call_id="c1", tool_name="t", timestamp=2),
        ]
    )
    params = build_params(make_model(), ctx, {}, detect_compat(make_model()), "short")
    assert params["tools"] == []


def test_no_tools_field_without_tools_or_history():
    assert "tools" not in params_for(make_model())


def test_zai_tool_stream_is_sent_when_enabled():
    compat = replace(detect_compat(make_model()), zai_tool_stream=True)
    ctx = Context(messages=[user_text("hi")], tools=[Tool(name="t", description="d", parameters={})])
    params = build_params(make_model(), ctx, {}, compat, "short")
    assert params["tool_stream"] is True


def test_kimi_deferred_tools_are_filtered_from_the_initial_request():
    compat = replace(detect_compat(make_model()), deferred_tools_mode="kimi")
    ctx = Context(
        messages=[
            ToolResultMessage(
                tool_call_id="c1", tool_name="t", added_tool_names=["later"], timestamp=1
            )
        ],
        tools=[Tool(name="now", description="d", parameters={}), Tool(name="later", description="d", parameters={})],
    )
    params = build_params(make_model(), ctx, {}, compat, "short")
    assert [t["function"]["name"] for t in params["tools"]] == ["now"]


def test_tool_choice_is_passed_through():
    assert params_for(make_model(), {"tool_choice": "auto"})["tool_choice"] == "auto"


# --- 11 种 thinking 格式 ---


def test_zai_thinking_enabled_and_disabled():
    compat = replace(detect_compat(make_model()), thinking_format="zai", supports_reasoning_effort=False)
    assert params_for(make_model(), {"reasoning_effort": "medium"}, compat=compat)["thinking"] == {
        "type": "enabled",
        "clear_thinking": False,
    }
    assert params_for(make_model(), {}, compat=compat)["thinking"] == {"type": "disabled"}


def test_zai_adds_reasoning_effort_when_supported():
    compat = replace(detect_compat(make_model()), thinking_format="zai", supports_reasoning_effort=True)
    params = params_for(make_model(), {"reasoning_effort": "medium"}, compat=compat)
    assert params["reasoning_effort"] == "medium"


def test_qwen_uses_enable_thinking():
    compat = replace(detect_compat(make_model()), thinking_format="qwen")
    assert params_for(make_model(), {"reasoning_effort": "high"}, compat=compat)["enable_thinking"] is True
    assert params_for(make_model(), {}, compat=compat)["enable_thinking"] is False


def test_qwen_chat_template_sets_both_flags():
    compat = replace(detect_compat(make_model()), thinking_format="qwen-chat-template")
    params = params_for(make_model(), {"reasoning_effort": "high"}, compat=compat)
    assert params["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}


def test_chat_template_resolves_var_placeholders():
    compat = replace(
        detect_compat(make_model()),
        thinking_format="chat-template",
        chat_template_kwargs={
            "enable_thinking": {"$var": "thinking.enabled"},
            "effort": {"$var": "thinking.effort"},
            "extra": "literal",
        },
    )
    params = params_for(make_model(), {"reasoning_effort": "medium"}, compat=compat)
    assert params["chat_template_kwargs"] == {
        "enable_thinking": True,
        "effort": "medium",
        "extra": "literal",
    }


def test_chat_template_omit_when_off_drops_the_key():
    compat = replace(
        detect_compat(make_model()),
        thinking_format="chat-template",
        chat_template_kwargs={"effort": {"$var": "thinking.effort", "omitWhenOff": True}},
    )
    assert "chat_template_kwargs" not in params_for(make_model(), {}, compat=compat)


def test_baseten_uses_chat_template_args():
    compat = replace(
        detect_compat(make_model()),
        thinking_format="baseten",
        chat_template_args={"thinking": {"$var": "thinking.enabled"}},
        supports_reasoning_effort=True,
    )
    params = params_for(make_model(), {"reasoning_effort": "low"}, compat=compat)
    assert params["chat_template_args"] == {"thinking": True}
    assert params["reasoning_effort"] == "low"


def test_baseten_uses_the_off_mapping_when_no_effort():
    compat = replace(
        detect_compat(make_model()), thinking_format="baseten", supports_reasoning_effort=True
    )
    model = make_model(level_map={"off": "none"})
    assert params_for(model, {}, compat=compat)["reasoning_effort"] == "none"


def test_deepseek_thinking_enabled_and_disabled():
    compat = replace(detect_compat(make_model()), thinking_format="deepseek", supports_reasoning_effort=True)
    params = params_for(make_model(), {"reasoning_effort": "high"}, compat=compat)
    assert params["thinking"] == {"type": "enabled"}
    assert params["reasoning_effort"] == "high"

    assert params_for(make_model(), {}, compat=compat)["thinking"] == {"type": "disabled"}


def test_deepseek_skips_disabled_when_off_is_unsupported():
    compat = replace(detect_compat(make_model()), thinking_format="deepseek")
    model = make_model(level_map={"off": None})
    assert "thinking" not in params_for(model, {}, compat=compat)


def test_openrouter_uses_a_nested_reasoning_object():
    compat = replace(detect_compat(make_model()), thinking_format="openrouter")
    assert params_for(make_model(), {"reasoning_effort": "medium"}, compat=compat)["reasoning"] == {
        "effort": "medium"
    }
    assert params_for(make_model(), {}, compat=compat)["reasoning"] == {"effort": "none"}


def test_openrouter_uses_the_off_mapping_when_present():
    compat = replace(detect_compat(make_model()), thinking_format="openrouter")
    model = make_model(level_map={"off": "minimal"})
    assert params_for(model, {}, compat=compat)["reasoning"] == {"effort": "minimal"}


def test_ant_ling_only_sends_mapped_efforts():
    compat = replace(detect_compat(make_model()), thinking_format="ant-ling")
    model = make_model(level_map={"high": "high"})
    assert params_for(model, {"reasoning_effort": "high"}, compat=compat)["reasoning"] == {"effort": "high"}
    # 没有映射时什么都不发
    assert "reasoning" not in params_for(make_model(), {"reasoning_effort": "high"}, compat=compat)
    assert "reasoning" not in params_for(model, {}, compat=compat)


def test_together_uses_an_enabled_flag():
    compat = replace(detect_compat(make_model()), thinking_format="together", supports_reasoning_effort=True)
    params = params_for(make_model(), {"reasoning_effort": "medium"}, compat=compat)
    assert params["reasoning"] == {"enabled": True}
    assert params["reasoning_effort"] == "medium"
    assert params_for(make_model(), {}, compat=compat)["reasoning"] == {"enabled": False}


def test_string_thinking_uses_a_top_level_string():
    compat = replace(detect_compat(make_model()), thinking_format="string-thinking")
    assert params_for(make_model(), {"reasoning_effort": "high"}, compat=compat)["thinking"] == "high"
    assert params_for(make_model(), {}, compat=compat)["thinking"] == "none"


def test_openai_default_uses_reasoning_effort():
    compat = detect_compat(make_model())
    assert params_for(make_model(), {"reasoning_effort": "medium"}, compat=compat)["reasoning_effort"] == "medium"


def test_openai_default_uses_the_off_mapping_when_no_effort():
    model = make_model(level_map={"off": "minimal"})
    assert params_for(model, {})["reasoning_effort"] == "minimal"


def test_openai_default_sends_nothing_when_off_is_unmapped():
    assert "reasoning_effort" not in params_for(make_model(), {})


def test_thinking_is_skipped_for_non_reasoning_models():
    model = make_model(reasoning=False)
    assert "reasoning_effort" not in params_for(model, {"reasoning_effort": "high"})


def test_level_map_translates_the_effort_value():
    model = make_model(level_map={"medium": "balanced"})
    assert params_for(model, {"reasoning_effort": "medium"})["reasoning_effort"] == "balanced"


# --- thinking token budget ---


def test_thinking_token_budget_is_capped_to_leave_answer_room():
    compat = replace(detect_compat(make_model()), supports_thinking_token_budget=True)
    params = params_for(make_model(), {"reasoning_effort": "high", "max_tokens": 4000}, compat=compat)
    assert params["thinking_token_budget"] == 4000 - 1024


def test_thinking_token_budget_uses_the_level_default():
    compat = replace(detect_compat(make_model()), supports_thinking_token_budget=True)
    params = params_for(make_model(), {"reasoning_effort": "low", "max_tokens": 100000}, compat=compat)
    assert params["thinking_token_budget"] == 2048


def test_thinking_token_budget_respects_custom_budgets():
    compat = replace(detect_compat(make_model()), supports_thinking_token_budget=True)
    params = params_for(
        make_model(),
        {"reasoning_effort": "low", "max_tokens": 100000, "thinking_budgets": {"low": 77}},
        compat=compat,
    )
    assert params["thinking_token_budget"] == 77


def test_thinking_token_budget_is_omitted_when_there_is_no_room():
    compat = replace(detect_compat(make_model()), supports_thinking_token_budget=True)
    params = params_for(make_model(), {"reasoning_effort": "high", "max_tokens": 500}, compat=compat)
    assert "thinking_token_budget" not in params


def test_thinking_token_budget_is_omitted_without_the_flag():
    params = params_for(make_model(), {"reasoning_effort": "high", "max_tokens": 4000})
    assert "thinking_token_budget" not in params


# --- 路由与采样参数 ---


def test_open_router_routing_is_only_sent_from_explicit_compat():
    assert "provider" not in params_for(make_model(provider="openrouter"))

    model = make_model(provider="openrouter", compat={"openRouterRouting": {"order": ["anthropic"]}})
    assert params_for(model)["provider"] == {"order": ["anthropic"]}


def test_vercel_gateway_routing_requires_only_or_order():
    model = make_model(compat={"vercelGatewayRouting": {"only": ["bedrock"]}})
    assert params_for(model)["providerOptions"] == {"gateway": {"only": ["bedrock"]}}

    model = make_model(compat={"vercelGatewayRouting": {}})
    assert "providerOptions" not in params_for(model)


def test_sampling_params_are_merged_last_and_can_override():
    params = params_for(make_model(), {"temperature": 0.1, "sampling_params": {"temperature": 0.9, "top_k": 5}})
    assert params["temperature"] == 0.9
    assert params["top_k"] == 5


# --- 缓存保留策略 ---


def test_resolve_cache_retention_prefers_explicit_then_env():
    assert resolve_cache_retention("none", None) == "none"
    assert resolve_cache_retention(None, {"PI_CACHE_RETENTION": "long"}) == "long"
    assert resolve_cache_retention(None, {}) == "short"


def test_anthropic_cache_control_is_applied_to_messages():
    compat = replace(detect_compat(make_model()), cache_control_format="anthropic")
    ctx = Context(system_prompt="sys", messages=[user_text("hi")])
    params = build_params(make_model(), ctx, {}, compat, "short")
    assert params["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
