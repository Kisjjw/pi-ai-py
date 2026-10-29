from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pi_ai.api.anthropic_messages_compat import force_adaptive_thinking, get_anthropic_compat
from pi_ai.api.anthropic_messages_messages import (
    convert_messages,
    convert_tools,
    normalize_tool_call_id,
    to_claude_code_name,
)
from pi_ai.api.transform_messages import transform_messages
from pi_ai.types import Context, Model
from pi_ai.utils.deferred_tools import split_deferred_tools
from pi_ai.utils.provider_env import get_provider_env_value
from pi_ai.utils.sanitize import sanitize_surrogates

CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

DEFAULT_THINKING_BUDGET_TOKENS = 1024
DEFAULT_THINKING_DISPLAY = "summarized"


def resolve_cache_retention(
    cache_retention: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def get_cache_control(
    model: Model,
    cache_retention: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    retention = resolve_cache_retention(cache_retention, env)
    if retention == "none":
        return retention, None
    if retention == "long" and get_anthropic_compat(model).supports_long_cache_retention:
        return retention, {"type": "ephemeral", "ttl": "1h"}
    return retention, {"type": "ephemeral"}


def map_thinking_level_to_effort(model: Model, level: str | None) -> str:
    """把统一的 thinking 级别映射成 Anthropic 的 effort。

    effort "max" 在所有自适应思考模型上都可用，而原生的 "xhigh" 只有
    Opus 4.7/4.8、Sonnet 5、Fable 5 支持。
    """
    mapped = (model.thinking_level_map or {}).get(level) if level else None
    if isinstance(mapped, str):
        return mapped

    if level in ("minimal", "low"):
        return "low"
    if level == "medium":
        return "medium"
    return "high"


def should_use_fine_grained_tool_streaming_beta(model: Model, context: Context) -> bool:
    return bool(context.tools) and not get_anthropic_compat(model).supports_eager_tool_input_streaming


def build_params(
    model: Model,
    context: Context,
    is_oauth: bool,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造发往 /v1/messages 的请求体。

    移植自 TS 的 anthropic-messages.ts:939-1074。
    """
    options = options or {}
    _retention, cache_control = get_cache_control(
        model, options.get("cache_retention"), options.get("env")
    )
    compat = get_anthropic_compat(model)

    transformed = transform_messages(
        context.messages, model, lambda call_id, _model, _source: normalize_tool_call_id(call_id)
    )
    normalize_tool_name = to_claude_code_name if is_oauth else (lambda name: name)

    tool_context = context.model_copy(update={"messages": list(transformed)})
    immediate_tools, deferred_map = split_deferred_tools(
        tool_context, compat.supports_tool_references, normalize_tool_name
    )
    deferred_tools = list(deferred_map.values())
    # 全部工具都被判为延迟加载时请求里会一个工具都没有，此时整体提升为立即工具。
    if not immediate_tools and deferred_tools:
        immediate_tools, deferred_tools = deferred_tools, []
    deferred_tool_names = {normalize_tool_name(tool.name) for tool in deferred_tools}

    params: dict[str, Any] = {
        "model": model.id,
        "messages": convert_messages(
            transformed,
            is_oauth,
            cache_control,
            compat.allow_empty_signature,
            deferred_tool_names,
            normalize_tool_name,
        ),
        "max_tokens": options.get("max_tokens") or model.max_tokens,
        "stream": True,
    }

    _apply_system(params, context, is_oauth, cache_control)

    # temperature 与扩展思考互斥，且 Claude Opus 4.7+ 不接受非默认值。
    thinking_enabled = options.get("thinking_enabled")
    if options.get("temperature") is not None and not thinking_enabled and compat.supports_temperature:
        params["temperature"] = options["temperature"]

    if immediate_tools or deferred_tools:
        params["tools"] = [
            *convert_tools(
                immediate_tools,
                is_oauth,
                compat.supports_eager_tool_input_streaming,
                compat.supports_strict_tools,
                cache_control if compat.supports_cache_control_on_tools else None,
            ),
            *convert_tools(
                deferred_tools,
                is_oauth,
                compat.supports_eager_tool_input_streaming,
                compat.supports_strict_tools,
                None,
                True,
            ),
        ]

    _apply_thinking(params, model, options)

    metadata = options.get("metadata")
    if metadata:
        user_id = metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    tool_choice = options.get("tool_choice")
    if tool_choice:
        params["tool_choice"] = {"type": tool_choice} if isinstance(tool_choice, str) else tool_choice

    return params


def _apply_system(
    params: dict[str, Any],
    context: Context,
    is_oauth: bool,
    cache_control: dict[str, Any] | None,
) -> None:
    """OAuth 请求必须带上 Claude Code 的身份块，否则会被拒。"""

    def text_block(text: str) -> dict[str, Any]:
        block: dict[str, Any] = {"type": "text", "text": text}
        if cache_control:
            block["cache_control"] = cache_control
        return block

    if is_oauth:
        system = [text_block(CLAUDE_CODE_IDENTITY)]
        if context.system_prompt:
            system.append(text_block(sanitize_surrogates(context.system_prompt)))
        params["system"] = system
    elif context.system_prompt:
        params["system"] = [text_block(sanitize_surrogates(context.system_prompt))]


def _apply_thinking(params: dict[str, Any], model: Model, options: Mapping[str, Any]) -> None:
    if not model.reasoning:
        return

    thinking_enabled = options.get("thinking_enabled")

    if thinking_enabled:
        # 默认 "summarized"，让 Opus 4.7 与 Mythos Preview 的行为与更早的
        # Claude 4 模型一致（后者的 API 默认也是 summarized）。
        display = options.get("thinking_display") or DEFAULT_THINKING_DISPLAY

        if force_adaptive_thinking(model):
            params["thinking"] = {"type": "adaptive", "display": display}
            effort = options.get("effort")
            if effort:
                params["output_config"] = {"effort": effort}
        else:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": options.get("thinking_budget_tokens") or DEFAULT_THINKING_BUDGET_TOKENS,
                "display": display,
            }
        return

    off_mode = (model.thinking_level_map or {}).get("off", "disabled")
    if thinking_enabled is False and off_mode is not None:
        # Sonnet 5.5 用 between_tools 关闭前置思考，不接受 disabled，
        # 且该模式不能携带 display 或 budget_tokens。
        params["thinking"] = {"type": "between_tools" if off_mode == "between_tools" else "disabled"}
