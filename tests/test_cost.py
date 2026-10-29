import pytest

from pi_ai.catalog import load_provider_models
from pi_ai.cost import calculate_cost, clamp_thinking_level, get_supported_thinking_levels
from pi_ai.types import Model, ModelCost, ModelCostTier, Usage


def make_model(**overrides) -> Model:
    defaults = dict(
        id="m",
        name="M",
        api="openai-completions",
        provider="p",
        base_url="http://x/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        context_window=200000,
        max_tokens=8192,
    )
    defaults.update(overrides)
    return Model(**defaults)


def test_cost_is_per_million_tokens():
    usage = Usage(input=1_000_000, output=1_000_000)
    cost = calculate_cost(make_model(), usage)
    assert cost.input == 3.0
    assert cost.output == 15.0
    assert cost.total == 18.0


def test_cache_read_and_write_are_priced_separately():
    usage = Usage(cache_read=1_000_000, cache_write=1_000_000)
    cost = calculate_cost(make_model(), usage)
    assert cost.cache_read == 0.3
    assert cost.cache_write == 3.75


def test_one_hour_cache_writes_are_billed_at_double_base_input():
    usage = Usage(cache_write=1_000_000, cache_write_1h=1_000_000)
    cost = calculate_cost(make_model(), usage)
    assert cost.cache_write == 6.0


def test_mixed_cache_write_retentions_split_the_rate():
    usage = Usage(cache_write=2_000_000, cache_write_1h=1_000_000)
    cost = calculate_cost(make_model(), usage)
    assert cost.cache_write == 3.75 + 6.0


def test_highest_matching_tier_applies_to_whole_request():
    model = make_model(
        cost=ModelCost(
            input=3.0,
            output=15.0,
            cache_read=0.3,
            cache_write=3.75,
            tiers=[
                ModelCostTier(input_tokens_above=200_000, input=6.0, output=22.5, cache_read=0.6, cache_write=7.5),
                ModelCostTier(input_tokens_above=1_000_000, input=9.0, output=30.0, cache_read=0.9, cache_write=11.25),
            ],
        )
    )
    usage = Usage(input=1_500_000, output=1_000_000)
    cost = calculate_cost(model, usage)
    assert cost.input == 13.5
    assert cost.output == 30.0


def test_tier_threshold_counts_cache_tokens_as_input():
    model = make_model(
        cost=ModelCost(
            input=3.0,
            output=15.0,
            cache_read=0.3,
            cache_write=3.75,
            tiers=[ModelCostTier(input_tokens_above=200_000, input=6.0, output=22.5, cache_read=0.6, cache_write=7.5)],
        )
    )
    usage = Usage(input=100_000, cache_read=150_000)
    cost = calculate_cost(model, usage)
    assert cost.input == 0.6


def test_non_reasoning_model_supports_only_off():
    assert get_supported_thinking_levels(make_model()) == ["off"]


def test_reasoning_model_exposes_standard_levels_but_not_optin_ones():
    levels = get_supported_thinking_levels(make_model(reasoning=True))
    assert levels == ["off", "minimal", "low", "medium", "high"]


def test_null_map_entry_removes_a_level():
    model = make_model(reasoning=True, thinking_level_map={"minimal": None, "low": None})
    assert get_supported_thinking_levels(model) == ["off", "medium", "high"]


def test_optin_levels_require_an_explicit_map_entry():
    model = make_model(reasoning=True, thinking_level_map={"xhigh": "xhigh"})
    assert get_supported_thinking_levels(model) == ["off", "minimal", "low", "medium", "high", "xhigh"]


def test_clamp_returns_level_when_supported():
    assert clamp_thinking_level(make_model(reasoning=True), "medium") == "medium"


def test_clamp_walks_upward_to_the_next_supported_level():
    model = make_model(reasoning=True, thinking_level_map={"minimal": None, "low": None})
    assert clamp_thinking_level(model, "low") == "medium"


def test_clamp_falls_back_downward_when_nothing_higher_exists():
    model = make_model(reasoning=True, thinking_level_map={"xhigh": None, "high": None})
    assert clamp_thinking_level(model, "high") == "medium"


def test_clamp_on_non_reasoning_model_yields_off():
    assert clamp_thinking_level(make_model(), "high") == "off"


@pytest.mark.parametrize(
    "provider,model_id,threshold,base_output,long_output",
    [
        ("openai", "gpt-6-astra", 272_000, 50, 75),
        ("openai", "gpt-6-sol", 272_000, 10, 15),
        ("openai", "gpt-6.1-sol", 272_000, 10, 15),
        ("openai", "gpt-6.1-terra", 272_000, 5, 7.5),
        ("openai", "gpt-6-luna", 272_000, 0.5, 0.75),
        ("anthropic", "claude-haiku-5-5", 100_000, 0.5, 2.5),
    ],
)
def test_current_model_long_context_prices_count_cache_and_apply_after_threshold(
    provider: str, model_id: str, threshold: int, base_output: float, long_output: float
) -> None:
    model = load_provider_models(provider)[model_id]
    at_threshold = Usage(input=threshold - 20_000, cache_read=10_000, cache_write=10_000, output=1000)
    above_threshold = at_threshold.model_copy(update={"cache_read": 10_001})

    assert calculate_cost(model, at_threshold).output == pytest.approx(base_output / 1000)
    assert calculate_cost(model, above_threshold).output == pytest.approx(long_output / 1000)
