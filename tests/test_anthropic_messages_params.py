import pytest

from pi_ai.api.anthropic_messages_params import (
    build_params,
    get_cache_control,
    map_thinking_level_to_effort,
    should_use_fine_grained_tool_streaming_beta,
)
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    user_text,
)


def make_model(
    *,
    provider="anthropic",
    model_id="claude-sonnet-4-5",
    reasoning=True,
    level_map=None,
    compat=None,
    max_tokens=8192,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="anthropic-messages",
        provider=provider,
        base_url="https://api.anthropic.com",
        reasoning=reasoning,
        thinking_level_map=level_map,
        input=["text"],
        cost=ModelCost(),
        context_window=200000,
        max_tokens=max_tokens,
        compat=compat,
    )


CTX = Context(messages=[user_text("hi")])


def tool(name="Read") -> Tool:
    return Tool(name=name, description="d", parameters={"type": "object", "properties": {}})


# --- 缓存 ---


def test_cache_control_by_retention():
    model = make_model()
    assert get_cache_control(model, "none") == ("none", None)
    assert get_cache_control(model, "short") == ("short", {"type": "ephemeral"})
    assert get_cache_control(model, "long") == ("long", {"type": "ephemeral", "ttl": "1h"})


def test_long_retention_degrades_when_unsupported():
    model = make_model(compat={"supportsLongCacheRetention": False})
    assert get_cache_control(model, "long") == ("long", {"type": "ephemeral"})


def test_env_can_request_long_retention():
    model = make_model()
    assert get_cache_control(model, None, {"PI_CACHE_RETENTION": "long"})[0] == "long"


# --- effort 映射 ---


@pytest.mark.parametrize(
    "level,expected",
    [("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"), (None, "high"), ("max", "high")],
)
def test_thinking_level_to_effort_defaults(level, expected):
    assert map_thinking_level_to_effort(make_model(), level) == expected


def test_thinking_level_map_overrides_the_effort():
    model = make_model(level_map={"medium": "xhigh"})
    assert map_thinking_level_to_effort(model, "medium") == "xhigh"


# --- 基础字段 ---


def test_core_fields():
    params = build_params(make_model(), CTX, False)
    assert params["model"] == "claude-sonnet-4-5"
    assert params["stream"] is True
    assert params["max_tokens"] == 8192
    assert params["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}
    ]


def test_explicit_max_tokens_wins():
    params = build_params(make_model(), CTX, False, {"max_tokens": 100})
    assert params["max_tokens"] == 100


# --- system ---


def test_system_prompt_is_wrapped_in_a_text_block():
    ctx = Context(system_prompt="Be brief.", messages=[user_text("hi")])
    params = build_params(make_model(), ctx, False)
    assert params["system"] == [
        {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}
    ]


def test_no_system_field_without_a_prompt():
    assert "system" not in build_params(make_model(), CTX, False)


def test_oauth_prepends_the_claude_code_identity():
    ctx = Context(system_prompt="Be brief.", messages=[user_text("hi")])
    params = build_params(make_model(), ctx, True)
    assert params["system"][0]["text"] == "You are Claude Code, Anthropic's official CLI for Claude."
    assert params["system"][1]["text"] == "Be brief."


def test_oauth_sends_the_identity_even_without_a_user_prompt():
    params = build_params(make_model(), CTX, True)
    assert len(params["system"]) == 1
    assert "Claude Code" in params["system"][0]["text"]


def test_system_blocks_have_no_cache_control_when_retention_is_none():
    ctx = Context(system_prompt="Be brief.", messages=[user_text("hi")])
    params = build_params(make_model(), ctx, False, {"cache_retention": "none"})
    assert params["system"] == [{"type": "text", "text": "Be brief."}]


# --- temperature ---


def test_temperature_is_sent_when_thinking_is_off():
    params = build_params(make_model(), CTX, False, {"temperature": 0.5})
    assert params["temperature"] == 0.5


def test_temperature_is_dropped_when_thinking_is_enabled():
    params = build_params(make_model(), CTX, False, {"temperature": 0.5, "thinking_enabled": True})
    assert "temperature" not in params


def test_temperature_is_dropped_when_unsupported():
    model = make_model(compat={"supportsTemperature": False})
    params = build_params(model, CTX, False, {"temperature": 0.5})
    assert "temperature" not in params


# --- thinking ---


def test_budget_thinking_for_regular_models():
    params = build_params(make_model(), CTX, False, {"thinking_enabled": True, "thinking_budget_tokens": 4096})
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 4096, "display": "summarized"}


def test_budget_thinking_defaults_to_1024():
    params = build_params(make_model(), CTX, False, {"thinking_enabled": True})
    assert params["thinking"]["budget_tokens"] == 1024


def test_thinking_display_can_be_omitted():
    params = build_params(
        make_model(), CTX, False, {"thinking_enabled": True, "thinking_display": "omitted"}
    )
    assert params["thinking"]["display"] == "omitted"


def test_adaptive_thinking_uses_output_config():
    model = make_model(compat={"forceAdaptiveThinking": True})
    params = build_params(model, CTX, False, {"thinking_enabled": True, "effort": "high"})
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"] == {"effort": "high"}


def test_adaptive_thinking_without_effort_sends_no_output_config():
    model = make_model(compat={"forceAdaptiveThinking": True})
    params = build_params(model, CTX, False, {"thinking_enabled": True})
    assert "output_config" not in params


def test_explicitly_disabled_thinking_is_sent():
    params = build_params(make_model(), CTX, False, {"thinking_enabled": False})
    assert params["thinking"] == {"type": "disabled"}


def test_disabled_thinking_is_skipped_when_off_is_unsupported():
    model = make_model(level_map={"off": None})
    params = build_params(model, CTX, False, {"thinking_enabled": False})
    assert "thinking" not in params


def test_non_reasoning_models_never_get_a_thinking_field():
    model = make_model(reasoning=False)
    assert "thinking" not in build_params(model, CTX, False, {"thinking_enabled": True})


def test_omitted_thinking_enabled_sends_nothing():
    assert "thinking" not in build_params(make_model(), CTX, False, {})


# --- 工具 ---


def test_tools_are_converted():
    ctx = Context(messages=[user_text("hi")], tools=[tool()])
    params = build_params(make_model(), ctx, False)
    assert params["tools"][0]["name"] == "Read"


def test_no_tools_field_without_tools():
    assert "tools" not in build_params(make_model(), CTX, False)


def test_deferred_tools_are_marked_and_excluded_from_cache_control():
    ctx = Context(
        messages=[
            AssistantMessage(
                content=[ToolCall(id="c1", name="Read", arguments={})],
                api="anthropic-messages",
                provider="anthropic",
                model="claude-sonnet-4-5",
                stop_reason="toolUse",
                timestamp=1,
            ),
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="Read",
                content=[TextContent(text="ok")],
                added_tool_names=["Later"],
                timestamp=2,
            ),
        ],
        tools=[tool("Read"), tool("Later")],
    )
    params = build_params(make_model(), ctx, False)
    by_name = {t["name"]: t for t in params["tools"]}
    assert by_name["Later"]["defer_loading"] is True
    assert "defer_loading" not in by_name["Read"]
    assert "cache_control" not in by_name["Later"]


def test_all_deferred_tools_get_promoted_to_immediate():
    """否则请求里会一个工具都没有。"""
    ctx = Context(
        messages=[
            ToolResultMessage(
                tool_call_id="c1", tool_name="t", added_tool_names=["Later"], timestamp=1
            )
        ],
        tools=[tool("Later")],
    )
    params = build_params(make_model(), ctx, False)
    assert len(params["tools"]) == 1
    assert "defer_loading" not in params["tools"][0]


def test_tool_references_are_disabled_for_incapable_models():
    ctx = Context(
        messages=[
            ToolResultMessage(
                tool_call_id="c1", tool_name="t", added_tool_names=["Later"], timestamp=1
            )
        ],
        tools=[tool("Read"), tool("Later")],
    )
    params = build_params(make_model(model_id="claude-haiku-4-5"), ctx, False)
    assert all("defer_loading" not in t for t in params["tools"])


def test_cache_control_lands_on_the_last_immediate_tool():
    ctx = Context(messages=[user_text("hi")], tools=[tool("a"), tool("b")])
    params = build_params(make_model(), ctx, False)
    assert "cache_control" not in params["tools"][0]
    assert params["tools"][1]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_on_tools_can_be_disabled():
    model = make_model(compat={"supportsCacheControlOnTools": False})
    ctx = Context(messages=[user_text("hi")], tools=[tool()])
    params = build_params(model, ctx, False)
    assert "cache_control" not in params["tools"][0]


# --- metadata 与 tool_choice ---


def test_metadata_only_forwards_a_string_user_id():
    assert build_params(make_model(), CTX, False, {"metadata": {"user_id": "u1"}})["metadata"] == {
        "user_id": "u1"
    }
    assert "metadata" not in build_params(make_model(), CTX, False, {"metadata": {"user_id": 1}})
    assert "metadata" not in build_params(make_model(), CTX, False, {"metadata": {"other": "x"}})


def test_string_tool_choice_is_wrapped():
    assert build_params(make_model(), CTX, False, {"tool_choice": "any"})["tool_choice"] == {"type": "any"}


def test_object_tool_choice_passes_through():
    choice = {"type": "tool", "name": "Read"}
    assert build_params(make_model(), CTX, False, {"tool_choice": choice})["tool_choice"] == choice


# --- beta 判定 ---


def test_fine_grained_beta_requires_tools_and_no_eager_support():
    model = make_model(compat={"supportsEagerToolInputStreaming": False})
    ctx = Context(messages=[user_text("hi")], tools=[tool()])
    assert should_use_fine_grained_tool_streaming_beta(model, ctx) is True
    assert should_use_fine_grained_tool_streaming_beta(model, CTX) is False
    assert should_use_fine_grained_tool_streaming_beta(make_model(), ctx) is False
