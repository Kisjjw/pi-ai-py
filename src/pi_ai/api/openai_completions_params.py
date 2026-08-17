from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pi_ai.api.constrained_sampling import create_grammar_tool_input_properties
from pi_ai.api.openai_completions_compat import ResolvedCompat, get_compat
from pi_ai.api.openai_completions_messages import (
    apply_anthropic_cache_control,
    convert_messages,
    convert_tools,
    get_compat_cache_control,
    get_deferred_tool_names,
    has_tool_history,
)
from pi_ai.api.openai_prompt_cache import clamp_openai_prompt_cache_key
from pi_ai.api.simple_options import MIN_ANSWER_TOKENS, clamp_reasoning
from pi_ai.types import Context, Model
from pi_ai.utils.provider_env import get_provider_env_value

_DEFAULT_THINKING_TOKEN_BUDGETS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}

_MISSING = object()


def resolve_cache_retention(
    cache_retention: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _mapped_level(model: Model, level: str | None) -> Any:
    """读取 thinking_level_map，区分"键不存在"与"值为 None"。

    键不存在表示用原值，值为 None 表示该级别不受支持。
    """
    if level is None:
        return _MISSING
    level_map = model.thinking_level_map or {}
    return level_map.get(level, _MISSING)


def _off_level(model: Model) -> Any:
    level_map = model.thinking_level_map or {}
    return level_map.get("off", _MISSING)


def _effort_or_mapped(model: Model, effort: str) -> Any:
    """有映射用映射值，没映射用原始 effort。"""
    mapped = _mapped_level(model, effort)
    return effort if mapped is _MISSING else mapped


def _resolve_chat_template_value(model: Model, effort: str | None, value: Any) -> Any:
    """求值单个 chat template 参数，返回 _MISSING 表示丢弃该键。"""
    if not isinstance(value, dict):
        return value

    if not effort and value.get("omitWhenOff"):
        return _MISSING
    if value.get("$var") == "thinking.enabled":
        return bool(effort)

    mapped = _mapped_level(model, effort) if effort else _off_level(model)
    if mapped is _MISSING:
        return effort if effort is not None else _MISSING
    return mapped if isinstance(mapped, str) else _MISSING


def build_chat_template_values(
    model: Model,
    effort: str | None,
    values: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    resolved: dict[str, Any] = {}
    for key, value in (values or {}).items():
        result = _resolve_chat_template_value(model, effort, value)
        if result is not _MISSING:
            resolved[key] = result
    return resolved or None


def _apply_thinking(
    params: dict[str, Any],
    model: Model,
    compat: ResolvedCompat,
    effort: str | None,
) -> None:
    """按 thinking_format 写入对应的思考控制字段。

    移植自 openai-completions.ts:748-845。这是一条 if/elif 链，顺序敏感。
    """
    fmt = compat.thinking_format
    reasoning = model.reasoning

    if fmt == "zai" and reasoning:
        params["thinking"] = {"type": "enabled", "clear_thinking": False} if effort else {"type": "disabled"}
        if effort and compat.supports_reasoning_effort:
            value = _effort_or_mapped(model, effort)
            if isinstance(value, str):
                params["reasoning_effort"] = value
        return

    if fmt == "qwen" and reasoning:
        params["enable_thinking"] = bool(effort)
        if effort and compat.supports_reasoning_effort:
            value = _effort_or_mapped(model, effort)
            if isinstance(value, str):
                params["reasoning_effort"] = value
        return

    if fmt == "qwen-chat-template" and reasoning:
        params["chat_template_kwargs"] = {"enable_thinking": bool(effort), "preserve_thinking": True}
        return

    if fmt == "chat-template" and reasoning:
        kwargs = build_chat_template_values(model, effort, compat.chat_template_kwargs)
        if kwargs:
            params["chat_template_kwargs"] = kwargs
        return

    if fmt == "baseten" and reasoning:
        args = build_chat_template_values(model, effort, compat.chat_template_args)
        if args:
            params["chat_template_args"] = args
        if compat.supports_reasoning_effort:
            mapped = _mapped_level(model, effort) if effort else _off_level(model)
            value = effort if mapped is _MISSING else mapped
            if isinstance(value, str):
                params["reasoning_effort"] = value
        return

    if fmt == "deepseek" and reasoning:
        if effort:
            params["thinking"] = {"type": "enabled"}
        elif _off_level(model) is not None:
            params["thinking"] = {"type": "disabled"}
        if effort and compat.supports_reasoning_effort:
            params["reasoning_effort"] = _effort_or_mapped(model, effort)
        return

    if fmt == "openrouter" and reasoning:
        # OpenRouter 用一个嵌套的 reasoning 对象在各厂商之间抹平差异。
        if effort:
            params["reasoning"] = {"effort": _effort_or_mapped(model, effort)}
        elif _off_level(model) is not None:
            off = _off_level(model)
            params["reasoning"] = {"effort": "none" if off is _MISSING else off}
        return

    if fmt == "ant-ling" and reasoning and effort:
        mapped = _mapped_level(model, effort)
        if isinstance(mapped, str):
            params["reasoning"] = {"effort": mapped}
        return

    if fmt == "together" and reasoning:
        params["reasoning"] = {"enabled": bool(effort)}
        if effort and compat.supports_reasoning_effort:
            params["reasoning_effort"] = _effort_or_mapped(model, effort)
        return

    if fmt == "string-thinking" and reasoning:
        if effort:
            params["thinking"] = _effort_or_mapped(model, effort)
        elif _off_level(model) is not None:
            off = _off_level(model)
            params["thinking"] = "none" if off is _MISSING else off
        return

    # 默认的 OpenAI 风格 reasoning_effort
    if effort and reasoning and compat.supports_reasoning_effort:
        params["reasoning_effort"] = _effort_or_mapped(model, effort)
    elif not effort and reasoning and compat.supports_reasoning_effort:
        off = _off_level(model)
        if isinstance(off, str):
            params["reasoning_effort"] = off


def _apply_thinking_token_budget(
    params: dict[str, Any],
    model: Model,
    compat: ResolvedCompat,
    effort: str | None,
    thinking_budgets: Mapping[str, int] | None,
) -> None:
    """vLLM 用顶层 thinking_token_budget 给推理设上限。

    与 thinking_format 无关：同一台服务器可以同时服务 zai、qwen、chat-template
    模型。这里推理和答案共享 max_tokens，不设上限的话推理阶段可能吃掉整个响应，
    既没有答案也没有工具调用。
    """
    if not (compat.supports_thinking_token_budget and effort and model.reasoning):
        return

    level = clamp_reasoning(effort)
    budgets = {**_DEFAULT_THINKING_TOKEN_BUDGETS, **(thinking_budgets or {})}
    if level not in budgets:
        return

    ceiling = params.get("max_tokens") or params.get("max_completion_tokens") or model.max_tokens
    # 始终给答案留出空间，否则这个预算会重现它本要防的问题。
    budget = min(budgets[level], max(0, ceiling - MIN_ANSWER_TOKENS))
    if budget > 0:
        params["thinking_token_budget"] = budget


def build_params(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
    compat: ResolvedCompat | None = None,
    cache_retention: str | None = None,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """构造发往 Chat Completions 端点的请求体。

    移植自 openai-completions.ts:681-890。
    """
    options = options or {}
    compat = compat if compat is not None else get_compat(model)
    if cache_retention is None:
        cache_retention = resolve_cache_retention(options.get("cache_retention"), options.get("env"))
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools, compat.supports_openai_grammar_tools
        )

    messages = convert_messages(model, context, compat, grammar_tool_input_properties)
    cache_control = get_compat_cache_control(compat, cache_retention)

    params: dict[str, Any] = {"model": model.id, "messages": messages, "stream": True}

    session_id = options.get("session_id")
    wants_openai_cache_key = "api.openai.com" in model.base_url and cache_retention != "none"
    wants_long_cache = cache_retention == "long" and compat.supports_long_cache_retention
    if wants_openai_cache_key or wants_long_cache:
        cache_key = clamp_openai_prompt_cache_key(session_id)
        if cache_key is not None:
            params["prompt_cache_key"] = cache_key
    if wants_long_cache:
        params["prompt_cache_retention"] = "24h"

    if compat.supports_usage_in_streaming is not False:
        params["stream_options"] = {"include_usage": True}

    if compat.supports_store:
        params["store"] = False

    max_tokens = options.get("max_tokens")
    if max_tokens:
        if compat.max_tokens_field == "max_tokens":
            params["max_tokens"] = max_tokens
        else:
            params["max_completion_tokens"] = max_tokens

    if options.get("temperature") is not None:
        params["temperature"] = options["temperature"]

    deferred_names = (
        get_deferred_tool_names(context.messages) if compat.deferred_tools_mode == "kimi" else set()
    )
    active_tools = [t for t in (context.tools or []) if t.name not in deferred_names]
    tools: list[dict[str, Any]] | None = None
    if active_tools:
        tools = convert_tools(active_tools, compat)
        params["tools"] = tools
        if compat.zai_tool_stream:
            params["tool_stream"] = True
    elif has_tool_history(context.messages):
        # 经 LiteLLM / 代理转发的 Anthropic 在历史里有工具调用时要求带 tools 字段
        tools = []
        params["tools"] = tools

    if cache_control:
        apply_anthropic_cache_control(messages, tools, cache_control)

    if options.get("tool_choice"):
        params["tool_choice"] = options["tool_choice"]

    effort = options.get("reasoning_effort")
    _apply_thinking(params, model, compat, effort)
    _apply_thinking_token_budget(params, model, compat, effort, options.get("thinking_budgets"))

    # 路由偏好只在 model.compat 显式给出时才发，推断出的空默认值不发。
    raw_compat = model.compat
    if raw_compat is not None:
        open_router_routing = getattr(raw_compat, "open_router_routing", None)
        if open_router_routing:
            params["provider"] = open_router_routing

        vercel_routing = getattr(raw_compat, "vercel_gateway_routing", None)
        if vercel_routing and (vercel_routing.get("only") or vercel_routing.get("order")):
            gateway: dict[str, Any] = {}
            if vercel_routing.get("only"):
                gateway["only"] = vercel_routing["only"]
            if vercel_routing.get("order"):
                gateway["order"] = vercel_routing["order"]
            params["providerOptions"] = {"gateway": gateway}

    # 放在最后，让自定义键可以覆盖上面的具名字段。
    if options.get("sampling_params"):
        params.update(options["sampling_params"])

    return params
