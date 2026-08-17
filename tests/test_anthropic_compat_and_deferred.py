import pytest

from pi_ai.api.anthropic_messages_compat import (
    default_supports_tool_references,
    force_adaptive_thinking,
    get_anthropic_compat,
)
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
from pi_ai.utils.deferred_tools import split_deferred_tools


def make_model(*, provider="anthropic", model_id="claude-sonnet-4-5", compat=None) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="anthropic-messages",
        provider=provider,
        base_url="https://api.anthropic.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=200000,
        max_tokens=8192,
        compat=compat,
    )


def tool(name: str) -> Tool:
    return Tool(name=name, description="d", parameters={"type": "object"})


# --- 延迟工具拆分 ---


def test_disabled_puts_everything_in_immediate():
    ctx = Context(messages=[user_text("hi")], tools=[tool("a"), tool("b")])
    immediate, deferred = split_deferred_tools(ctx, False)
    assert [t.name for t in immediate] == ["a", "b"]
    assert deferred == {}


def test_duplicate_tool_names_are_deduplicated():
    ctx = Context(messages=[], tools=[tool("a"), tool("a")])
    immediate, _ = split_deferred_tools(ctx, False)
    assert len(immediate) == 1


def test_tools_added_by_a_tool_result_become_deferred():
    ctx = Context(
        messages=[
            ToolResultMessage(tool_call_id="c1", tool_name="t", added_tool_names=["later"], timestamp=1)
        ],
        tools=[tool("now"), tool("later")],
    )
    immediate, deferred = split_deferred_tools(ctx, True)
    assert [t.name for t in immediate] == ["now"]
    assert set(deferred) == {"later"}


def test_already_used_tools_are_not_deferred():
    """模型已经调用过的工具说明它已经知道了，不再算延迟加载。"""
    ctx = Context(
        messages=[
            AssistantMessage(
                content=[ToolCall(id="c1", name="later", arguments={})],
                api="anthropic-messages",
                provider="anthropic",
                model="m",
                stop_reason="toolUse",
                timestamp=1,
            ),
            ToolResultMessage(
                tool_call_id="c1", tool_name="later", added_tool_names=["later"], timestamp=2
            ),
        ],
        tools=[tool("later")],
    )
    immediate, deferred = split_deferred_tools(ctx, True)
    assert [t.name for t in immediate] == ["later"]
    assert deferred == {}


def test_normalizer_is_applied_to_both_sides():
    ctx = Context(
        messages=[
            ToolResultMessage(tool_call_id="c1", tool_name="t", added_tool_names=["read"], timestamp=1)
        ],
        tools=[tool("Read")],
    )
    immediate, deferred = split_deferred_tools(ctx, True, lambda name: name.lower())
    assert immediate == []
    assert set(deferred) == {"read"}


def test_no_tools_yields_empty_results():
    immediate, deferred = split_deferred_tools(Context(messages=[]), True)
    assert immediate == []
    assert deferred == {}


# --- tool_reference 默认值 ---


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("claude-sonnet-4-5", True),
        ("claude-opus-4-5", True),
        ("claude-opus-5", True),
        ("claude-fable-5", True),
        ("claude-sonnet-4", False),
        ("claude-opus-4-1", False),
        ("claude-3-5-sonnet", False),
    ],
)
def test_tool_reference_default_by_version(model_id, expected):
    assert default_supports_tool_references(make_model(model_id=model_id)) is expected


def test_haiku_never_supports_tool_references():
    assert default_supports_tool_references(make_model(model_id="claude-haiku-4-5")) is False


def test_non_anthropic_providers_never_support_tool_references():
    assert default_supports_tool_references(make_model(provider="minimax", model_id="claude-opus-5")) is False


def test_date_suffix_is_not_treated_as_a_minor_version():
    """claude-opus-4-20250514 的第二段是日期戳，不是次版本号。

    当成次版本号的话 20250514 >= 5 会让它错误地支持 tool_reference。
    """
    assert default_supports_tool_references(make_model(model_id="claude-opus-4-20250514")) is False
    assert default_supports_tool_references(make_model(model_id="claude-sonnet-4-5-20250929")) is True


def test_unparseable_model_ids_return_false():
    assert default_supports_tool_references(make_model(model_id="some-custom-model")) is False


# --- 兼容开关 ---


def test_defaults_without_explicit_compat():
    compat = get_anthropic_compat(make_model())
    assert compat.supports_eager_tool_input_streaming is True
    assert compat.supports_long_cache_retention is True
    assert compat.send_session_affinity_headers is False
    assert compat.supports_cache_control_on_tools is True
    assert compat.supports_temperature is True
    assert compat.allow_empty_signature is False
    assert compat.supports_strict_tools is False
    assert compat.supports_tool_references is True


def test_explicit_compat_overrides_defaults():
    model = make_model(
        compat={
            "supportsEagerToolInputStreaming": False,
            "supportsTemperature": False,
            "allowEmptySignature": True,
            "supportsStrictTools": True,
        }
    )
    compat = get_anthropic_compat(model)
    assert compat.supports_eager_tool_input_streaming is False
    assert compat.supports_temperature is False
    assert compat.allow_empty_signature is True
    assert compat.supports_strict_tools is True


def test_tool_references_can_be_forced_off_on_a_capable_model():
    model = make_model(compat={"supportsToolReferences": False})
    assert get_anthropic_compat(model).supports_tool_references is False


def test_compat_assigned_after_construction_is_still_parsed():
    model = make_model()
    model.compat = {"supportsTemperature": False}
    assert get_anthropic_compat(model).supports_temperature is False


def test_force_adaptive_thinking_requires_an_explicit_true():
    assert force_adaptive_thinking(make_model()) is False
    assert force_adaptive_thinking(make_model(compat={"forceAdaptiveThinking": False})) is False
    assert force_adaptive_thinking(make_model(compat={"forceAdaptiveThinking": True})) is True
