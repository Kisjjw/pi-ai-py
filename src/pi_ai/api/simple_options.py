from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pi_ai.types import Context, Model
from pi_ai.utils.estimate import estimate_context_tokens

CONTEXT_SAFETY_TOKENS = 4096
MIN_MAX_TOKENS = 1

#: 思考预算与答案共享同一个响应上限时，始终为答案保留的 token 数。
MIN_ANSWER_TOKENS = 1024

_DEFAULT_THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}

#: 从 build_base_options 透传到适配器的选项键，与 TS 的 StreamOptions 字段一一对应。
_PASSTHROUGH_KEYS = (
    "temperature",
    "signal",
    "api_key",
    "transport",
    "cache_retention",
    "session_id",
    "headers",
    "on_payload",
    "on_response",
    "timeout_ms",
    "websocket_connect_timeout_ms",
    "max_retries",
    "max_retry_delay_ms",
    "metadata",
    "env",
)

#: 适配器会读、但不在 TS 的 SimpleStreamOptions 里的选项键。
#:
#: TS 侧 streamSimple 的参数有类型约束，这些字段压根传不进去所以不显形；Python 全是
#: dict，调用方能传、也会理所当然地期待它生效，静默丢弃是最难查的一类问题。
#: 与 _PASSTHROUGH_KEYS 不同，这些只在调用方真的传了时才带过去：有的适配器用
#: options.get(key, 默认值) 读取，写成显式 None 会把默认值顶掉。
_OPTIONAL_KEYS = (
    # 中转站 / 自建网关：实际端点与模型目录里记的不一样
    "base_url",
    # openai-responses
    "service_tier",
    "reasoning_summary",
    # anthropic-messages
    "interleaved_thinking",
)


def clamp_max_tokens_to_context(model: Model, context: Context, max_tokens: int) -> int:
    """把请求的输出上限压到上下文窗口还能容纳的范围内。"""
    if model.context_window <= 0:
        return max(MIN_MAX_TOKENS, max_tokens)
    available = model.context_window - estimate_context_tokens(context).tokens - CONTEXT_SAFETY_TOKENS
    return min(max_tokens, max(MIN_MAX_TOKENS, available))


def build_base_options(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """把 SimpleStreamOptions 展开成适配器要的完整选项。"""
    options = options or {}

    request_sampling = options.get("sampling_params")
    sampling_params: dict[str, Any] | None = None
    if model.sampling_params or request_sampling:
        sampling_params = {**(model.sampling_params or {}), **(request_sampling or {})}

    base: dict[str, Any] = {key: options.get(key) for key in _PASSTHROUGH_KEYS}
    base.update({key: options[key] for key in _OPTIONAL_KEYS if key in options})
    base["sampling_params"] = sampling_params
    base["max_tokens"] = clamp_max_tokens_to_context(
        model, context, options.get("max_tokens") or model.max_tokens
    )
    base["api_key"] = api_key or options.get("api_key")
    return base


def clamp_reasoning(effort: str | None) -> str | None:
    """xhigh / max 是模型特定的选择性级别，对预算计算按 high 处理。"""
    return "high" if effort in ("xhigh", "max") else effort


def adjust_max_tokens_for_thinking(
    base_max_tokens: int | None,
    model_max_tokens: int,
    reasoning_level: str,
    custom_budgets: Mapping[str, int] | None = None,
) -> tuple[int, int]:
    """算出思考预算，并按需抬高输出上限。

    base_max_tokens 为 None 表示调用方没设上限，此时用模型上限并把思考塞进去。
    返回 (max_tokens, thinking_budget)。
    """
    budgets = {**_DEFAULT_THINKING_BUDGETS, **(custom_budgets or {})}
    level = clamp_reasoning(reasoning_level)
    thinking_budget = budgets[level] if level in budgets else _DEFAULT_THINKING_BUDGETS["medium"]

    if base_max_tokens is None:
        max_tokens = model_max_tokens
    else:
        max_tokens = min(base_max_tokens + thinking_budget, model_max_tokens)

    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - MIN_ANSWER_TOKENS)

    return max_tokens, thinking_budget
