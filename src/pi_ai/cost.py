from __future__ import annotations

from pi_ai.types import CostBreakdown, Model, ModelCostRates, Usage

EXTENDED_THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]

_OPT_IN_LEVELS = frozenset({"xhigh", "max"})

_MISSING = object()


def calculate_cost(model: Model, usage: Usage) -> CostBreakdown:
    """按模型定价算出费用，就地写入 usage.cost。

    分档定价按整次请求的输入总量（含缓存读写）选档，取满足条件的最高档。
    1 小时缓存写入按基础输入价的两倍计费（Anthropic 规则）。
    """
    input_tokens = usage.input + usage.cache_read + usage.cache_write

    rates: ModelCostRates = model.cost
    matched_threshold = -1
    for tier in model.cost.tiers or []:
        if input_tokens > tier.input_tokens_above and tier.input_tokens_above > matched_threshold:
            rates = tier
            matched_threshold = tier.input_tokens_above

    long_write = usage.cache_write_1h or 0
    short_write = usage.cache_write - long_write

    usage.cost.input = (rates.input / 1_000_000) * usage.input
    usage.cost.output = (rates.output / 1_000_000) * usage.output
    usage.cost.cache_read = (rates.cache_read / 1_000_000) * usage.cache_read
    usage.cost.cache_write = (rates.cache_write * short_write + rates.input * 2 * long_write) / 1_000_000
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    return usage.cost


def get_supported_thinking_levels(model: Model) -> list[str]:
    """列出模型支持的 thinking 级别。

    映射值为 None 表示该级别不支持；xhigh 与 max 是选择性开启的，
    必须在映射里显式给出才算支持。
    """
    if not model.reasoning:
        return ["off"]

    level_map = model.thinking_level_map or {}
    levels: list[str] = []
    for level in EXTENDED_THINKING_LEVELS:
        mapped = level_map.get(level, _MISSING)
        if mapped is None:
            continue
        if level in _OPT_IN_LEVELS and mapped is _MISSING:
            continue
        levels.append(level)
    return levels


def clamp_thinking_level(model: Model, level: str) -> str:
    """把请求的 thinking 级别裁剪到模型支持的最近级别，优先向上找。"""
    available = get_supported_thinking_levels(model)
    if level in available:
        return level

    if level not in EXTENDED_THINKING_LEVELS:
        return available[0] if available else "off"

    requested_index = EXTENDED_THINKING_LEVELS.index(level)
    for candidate in EXTENDED_THINKING_LEVELS[requested_index:]:
        if candidate in available:
            return candidate
    for candidate in reversed(EXTENDED_THINKING_LEVELS[:requested_index]):
        if candidate in available:
            return candidate
    return available[0] if available else "off"
