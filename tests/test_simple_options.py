from pi_ai.api.simple_options import (
    CONTEXT_SAFETY_TOKENS,
    MIN_ANSWER_TOKENS,
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_max_tokens_to_context,
    clamp_reasoning,
)
from pi_ai.types import Context, Model, ModelCost, user_text


def make_model(context_window=200000, max_tokens=8192, sampling_params=None) -> Model:
    return Model(
        id="m",
        name="M",
        api="openai-completions",
        provider="p",
        base_url="http://x/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=context_window,
        max_tokens=max_tokens,
        sampling_params=sampling_params,
    )


def test_clamp_leaves_room_for_context_and_safety_margin():
    ctx = Context(messages=[user_text("a" * 4000)])  # 约 1000 token
    clamped = clamp_max_tokens_to_context(make_model(context_window=10000), ctx, 8192)
    assert clamped == 10000 - 1000 - CONTEXT_SAFETY_TOKENS


def test_clamp_never_exceeds_the_requested_value():
    ctx = Context(messages=[user_text("hi")])
    assert clamp_max_tokens_to_context(make_model(context_window=1_000_000), ctx, 100) == 100


def test_clamp_floors_at_one_token():
    ctx = Context(messages=[user_text("a" * 40000)])
    assert clamp_max_tokens_to_context(make_model(context_window=1000), ctx, 8192) == 1


def test_clamp_skips_the_calculation_when_context_window_is_unknown():
    ctx = Context(messages=[user_text("a" * 40000)])
    assert clamp_max_tokens_to_context(make_model(context_window=0), ctx, 8192) == 8192
    assert clamp_max_tokens_to_context(make_model(context_window=0), ctx, 0) == 1


def test_clamp_reasoning_folds_optin_levels_to_high():
    assert clamp_reasoning("xhigh") == "high"
    assert clamp_reasoning("max") == "high"
    assert clamp_reasoning("medium") == "medium"
    assert clamp_reasoning(None) is None


def test_thinking_budget_defaults_per_level():
    for level, expected in [("minimal", 1024), ("low", 2048), ("medium", 8192), ("high", 16384)]:
        _max_tokens, budget = adjust_max_tokens_for_thinking(100000, 200000, level)
        assert budget == expected


def test_thinking_budget_uses_model_cap_when_no_explicit_cap():
    max_tokens, budget = adjust_max_tokens_for_thinking(None, 32000, "medium")
    assert max_tokens == 32000
    assert budget == 8192


def test_thinking_budget_adds_to_the_explicit_cap_but_respects_the_model_cap():
    max_tokens, _ = adjust_max_tokens_for_thinking(1000, 200000, "medium")
    assert max_tokens == 1000 + 8192

    max_tokens, _ = adjust_max_tokens_for_thinking(1000, 5000, "medium")
    assert max_tokens == 5000


def test_thinking_budget_always_leaves_room_for_the_answer():
    max_tokens, budget = adjust_max_tokens_for_thinking(None, 4000, "high")
    assert max_tokens == 4000
    assert budget == 4000 - MIN_ANSWER_TOKENS


def test_thinking_budget_cannot_go_negative():
    _max_tokens, budget = adjust_max_tokens_for_thinking(None, 500, "high")
    assert budget == 0


def test_custom_budgets_override_defaults():
    _max_tokens, budget = adjust_max_tokens_for_thinking(100000, 200000, "medium", {"medium": 999})
    assert budget == 999


def test_optin_levels_use_the_high_budget():
    _max_tokens, budget = adjust_max_tokens_for_thinking(100000, 200000, "xhigh")
    assert budget == 16384


def test_build_base_options_clamps_max_tokens_and_carries_options():
    ctx = Context(messages=[user_text("hi")])
    base = build_base_options(
        make_model(),
        ctx,
        {"temperature": 0.5, "session_id": "s1", "max_retries": 3},
        api_key="sk-1",
    )
    assert base["temperature"] == 0.5
    assert base["session_id"] == "s1"
    assert base["max_retries"] == 3
    assert base["api_key"] == "sk-1"
    assert base["max_tokens"] <= 8192


def test_build_base_options_defaults_max_tokens_to_the_model_cap():
    ctx = Context(messages=[user_text("hi")])
    base = build_base_options(make_model(max_tokens=555), ctx, None, None)
    assert base["max_tokens"] == 555


def test_build_base_options_forwards_adapter_specific_options():
    """base_url 之类的键不在 TS 的 SimpleStreamOptions 里，但 Python 调用方能传，必须生效。"""
    ctx = Context(messages=[user_text("hi")])
    base = build_base_options(
        make_model(),
        ctx,
        {
            "base_url": "https://relay.test/v1",
            "service_tier": "flex",
            "reasoning_summary": "detailed",
            "interleaved_thinking": False,
        },
        None,
    )
    assert base["base_url"] == "https://relay.test/v1"
    assert base["service_tier"] == "flex"
    assert base["reasoning_summary"] == "detailed"
    assert base["interleaved_thinking"] is False


def test_build_base_options_omits_adapter_options_that_were_not_passed():
    """不能写成显式 None：适配器用 options.get(key, 默认值) 读时会把默认值顶掉。"""
    ctx = Context(messages=[user_text("hi")])
    base = build_base_options(make_model(), ctx, {}, None)
    for key in ("base_url", "service_tier", "reasoning_summary", "interleaved_thinking"):
        assert key not in base


def test_build_base_options_merges_sampling_params_with_request_winning():
    ctx = Context(messages=[user_text("hi")])
    model = make_model(sampling_params={"top_p": 0.1, "top_k": 5})
    base = build_base_options(model, ctx, {"sampling_params": {"top_p": 0.9}}, None)
    assert base["sampling_params"] == {"top_p": 0.9, "top_k": 5}


def test_build_base_options_omits_sampling_params_when_neither_side_has_them():
    ctx = Context(messages=[user_text("hi")])
    base = build_base_options(make_model(), ctx, None, None)
    assert base.get("sampling_params") is None


def test_build_base_options_prefers_explicit_api_key_argument():
    ctx = Context(messages=[user_text("hi")])
    base = build_base_options(make_model(), ctx, {"api_key": "sk-opt"}, "sk-arg")
    assert base["api_key"] == "sk-arg"

    base = build_base_options(make_model(), ctx, {"api_key": "sk-opt"}, None)
    assert base["api_key"] == "sk-opt"
