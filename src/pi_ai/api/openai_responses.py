from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pi_ai.api.constrained_sampling import create_grammar_tool_input_properties
from pi_ai.api.github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from pi_ai.api.openai_prompt_cache import clamp_openai_prompt_cache_key
from pi_ai.api.openai_responses_shared import convert_responses_messages, convert_responses_tools
from pi_ai.api.openai_responses_stream import ResponsesStreamParser
from pi_ai.api.simple_options import build_base_options
from pi_ai.cost import clamp_thinking_level
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.http.client import OpenProviderStream, merge_headers, open_stream_post
from pi_ai.http.errors import ProviderHTTPError, format_provider_error, normalize_provider_error
from pi_ai.http.retry import retry_provider_request
from pi_ai.http.sse import iter_sse
from pi_ai.types import Context, DoneEvent, ErrorEvent, Model, OpenAIResponsesCompat, StartEvent, Usage
from pi_ai.utils.deferred_tools import split_deferred_tools
from pi_ai.utils.provider_env import get_provider_env_value

#: 只有这些 provider 的工具调用 id 保留 `callId|itemId` 复合结构。
OPENAI_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})

#: Responses 拒绝小于 16 的 max_output_tokens。
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16

_MISSING = object()


@dataclass(frozen=True)
class ResponsesResolvedCompat:
    supports_developer_role: bool
    session_affinity_format: str
    supports_long_cache_retention: bool
    supports_strict_mode: bool
    supports_openai_grammar_tools: bool
    supports_additional_tools: bool
    supports_tool_search: bool
    supports_explicit_prompt_cache_mode: bool


def detect_session_affinity_format(model: Model) -> str:
    return "openrouter" if model.provider == "openrouter" or "openrouter.ai" in model.base_url else "openai"


def get_compat(model: Model) -> ResponsesResolvedCompat:
    compat = model.compat
    if isinstance(compat, Mapping):
        compat = OpenAIResponsesCompat.model_validate(compat)

    def pick(name: str, fallback: Any) -> Any:
        if compat is None:
            return fallback
        value = getattr(compat, name, None)
        return fallback if value is None else value

    return ResponsesResolvedCompat(
        supports_developer_role=pick("supports_developer_role", True),
        session_affinity_format=pick("session_affinity_format", detect_session_affinity_format(model)),
        supports_long_cache_retention=pick("supports_long_cache_retention", True),
        supports_strict_mode=pick("supports_strict_mode", False),
        supports_openai_grammar_tools=pick("supports_openai_grammar_tools", False),
        supports_additional_tools=pick("supports_additional_tools", False),
        supports_tool_search=pick("supports_tool_search", False),
        supports_explicit_prompt_cache_mode=pick("supports_explicit_prompt_cache_mode", False),
    )


def resolve_cache_retention(
    cache_retention: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def build_request_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/responses"


def _has_header(headers: Mapping[str, str | None] | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected and value is not None and value.strip():
            return True
    return False


def resolve_api_key(
    provider: str,
    api_key: str | None,
    headers: Mapping[str, str | None] | None,
) -> str:
    if api_key:
        return api_key
    if _has_header(headers, "authorization") or _has_header(headers, "cf-aig-authorization"):
        return "unused"
    raise ProviderHTTPError(f"No API key for provider: {provider}")


def build_request_headers(
    model: Model,
    context: Context,
    compat: ResponsesResolvedCompat,
    api_key: str,
    options_headers: Mapping[str, str | None] | None = None,
    session_id: str | None = None,
) -> dict[str, str]:
    """拼请求头。

    与 Chat Completions 不同：这里的会话亲和不看开关，有 sessionId 就发。
    """
    headers: dict[str, str | None] = dict(model.headers or {})

    if model.provider == "github-copilot":
        headers.update(
            build_copilot_dynamic_headers(context.messages, has_copilot_vision_input(context.messages))
        )

    if session_id:
        if compat.session_affinity_format == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.session_affinity_format == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id

    merged = merge_headers(headers, options_headers)
    if not _has_header(merged, "authorization") and not _has_header(merged, "cf-aig-authorization"):
        merged["authorization"] = f"Bearer {api_key}"
    merged.setdefault("content-type", "application/json")
    merged.setdefault("accept", "text/event-stream")
    return merged


def get_service_tier_cost_multiplier(model: Model, service_tier: str | None) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.5 if model.id == "gpt-5.5" else 2.0
    return 1.0


def apply_service_tier_pricing(usage: Usage, service_tier: str | None, model: Model) -> None:
    multiplier = get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1.0:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write
    )


def build_params(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
    compat: ResponsesResolvedCompat | None = None,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    options = options or {}
    compat = compat if compat is not None else get_compat(model)
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools, compat.supports_openai_grammar_tools
        )

    if compat.supports_additional_tools:
        deferred_tools_mode: str | None = "additional-tools"
    elif compat.supports_tool_search:
        deferred_tools_mode = "tool-search"
    else:
        deferred_tools_mode = None

    immediate_tools, deferred_map = split_deferred_tools(context, deferred_tools_mode is not None)
    tool_options = {
        "supports_strict_mode": compat.supports_strict_mode,
        "supports_openai_grammar_tools": compat.supports_openai_grammar_tools,
    }

    messages = convert_responses_messages(
        model,
        context,
        OPENAI_TOOL_CALL_PROVIDERS,
        grammar_tool_input_properties=grammar_tool_input_properties,
        deferred_tools=deferred_map,
        deferred_tools_mode=deferred_tools_mode,
        tool_options=tool_options,
    )

    cache_retention = resolve_cache_retention(options.get("cache_retention"), options.get("env"))
    params: dict[str, Any] = {
        "model": model.id,
        "input": messages,
        "stream": True,
        "store": False,
    }

    if cache_retention != "none":
        cache_key = clamp_openai_prompt_cache_key(options.get("session_id"))
        if cache_key is not None:
            params["prompt_cache_key"] = cache_key
    if cache_retention == "long" and compat.supports_long_cache_retention:
        params["prompt_cache_retention"] = "24h"
    if cache_retention == "none" and compat.supports_explicit_prompt_cache_mode:
        params["prompt_cache_options"] = {"mode": "explicit"}

    max_tokens = options.get("max_tokens")
    if max_tokens:
        params["max_output_tokens"] = max(max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options.get("temperature") is not None:
        params["temperature"] = options["temperature"]

    if options.get("service_tier") is not None:
        params["service_tier"] = options["service_tier"]

    if immediate_tools:
        params["tools"] = convert_responses_tools(immediate_tools, **tool_options)

    if options.get("tool_choice") is not None:
        params["tool_choice"] = options["tool_choice"]

    _apply_reasoning(params, model, options)

    # 放在最后，让自定义键可以覆盖上面的具名字段。
    if options.get("sampling_params"):
        params.update(options["sampling_params"])

    return params


def _apply_reasoning(params: dict[str, Any], model: Model, options: Mapping[str, Any]) -> None:
    if not model.reasoning:
        return

    effort_option = options.get("reasoning_effort")
    summary_option = options.get("reasoning_summary")
    level_map = model.thinking_level_map or {}

    if effort_option or summary_option:
        effort = level_map.get(effort_option, effort_option) if effort_option else "medium"
        params["reasoning"] = {"effort": effort, "summary": summary_option or "auto"}
        params["include"] = ["reasoning.encrypted_content"]
    elif model.provider != "github-copilot" and level_map.get("off", _MISSING) is not None:
        off = level_map.get("off", _MISSING)
        params["reasoning"] = {"effort": "none" if off is _MISSING else off}

    if model.provider == "xai":
        params["include"] = ["reasoning.encrypted_content"]


def stream(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    options = dict(options or {})
    event_stream = AssistantMessageEventStream()

    async def run() -> None:
        compat = get_compat(model)
        grammar_properties = create_grammar_tool_input_properties(
            context.tools, compat.supports_openai_grammar_tools
        )
        parser = ResponsesStreamParser(
            model,
            grammar_tool_input_properties=dict(grammar_properties),
            service_tier=options.get("service_tier"),
            apply_service_tier_pricing=lambda usage, tier: apply_service_tier_pricing(usage, tier, model),
        )
        try:
            api_key = resolve_api_key(model.provider, options.get("api_key"), options.get("headers"))
            cache_retention = resolve_cache_retention(options.get("cache_retention"), options.get("env"))
            cache_session_id = None if cache_retention == "none" else options.get("session_id")

            params = build_params(model, context, options, compat, grammar_properties)
            on_payload = options.get("on_payload")
            if on_payload is not None:
                replaced = on_payload(params, model)
                if asyncio.iscoroutine(replaced):
                    replaced = await replaced
                if replaced is not None:
                    params = replaced

            headers = build_request_headers(
                model, context, compat, api_key, options.get("headers"), cache_session_id
            )
            url = build_request_url(options.get("base_url") or model.base_url)

            await consume_responses_stream(event_stream, parser, model, url, headers, params, options)
        except asyncio.CancelledError:
            emit_responses_error(event_stream, parser, aborted=True, error=None, prefix="OpenAI API error")
            raise
        except BaseException as error:  # noqa: BLE001 - 契约要求不抛出
            emit_responses_error(
                event_stream, parser, aborted=False, error=error, prefix="OpenAI API error"
            )

    task = asyncio.ensure_future(run())
    event_stream.bind_producer(task)
    return event_stream


async def consume_responses_stream(
    event_stream: AssistantMessageEventStream,
    parser: ResponsesStreamParser,
    model: Model,
    url: str,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
    options: Mapping[str, Any],
) -> None:
    """两个 Responses 适配器共用的消费逻辑。"""
    on_response = options.get("on_response")

    async def connect() -> OpenProviderStream:
        return await open_stream_post(
            url=url,
            headers=headers,
            json_body=params,
            timeout_ms=options.get("timeout_ms"),
        )

    # 重试只覆盖建立连接与首个响应；响应体一旦开始流式返回就不重放，与 TS 一致。
    opened = await retry_provider_request(
        connect,
        max_retries=options.get("max_retries") or 0,
        max_retry_delay_ms=options.get("max_retry_delay_ms"),
    )

    async with opened:
        if on_response is not None:
            result = on_response(opened.response, model)
            if asyncio.iscoroutine(result):
                await result

        event_stream.push(StartEvent(partial=parser.output))

        async for sse_event in iter_sse(opened.body):
            data = sse_event.data.strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            for event in parser.feed(payload):
                event_stream.push(event)

    if not parser.saw_terminal_event:
        raise ProviderHTTPError("OpenAI Responses stream ended before a terminal response event")

    output = parser.output
    if output.stop_reason == "pending":
        raise ProviderHTTPError("OpenAI Responses stream ended without a stop reason")
    if output.stop_reason == "aborted" or output.stop_reason == "error":  # noqa: PLR1714
        raise ProviderHTTPError(output.error_message or "An unknown error occurred")

    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
    event_stream.end()


def emit_responses_error(
    event_stream: AssistantMessageEventStream,
    parser: ResponsesStreamParser,
    *,
    aborted: bool,
    error: BaseException | None,
    prefix: str,
) -> None:
    output = parser.output
    output.stop_reason = "aborted" if aborted else "error"
    output.error_message = (
        "Request was aborted"
        if error is None
        else format_provider_error(normalize_provider_error(error), prefix)
    )
    event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
    event_stream.end()


def stream_simple(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    options = dict(options or {})
    base = build_base_options(model, context, options, options.get("api_key"))

    reasoning = options.get("reasoning")
    clamped = clamp_thinking_level(model, reasoning) if reasoning else None
    reasoning_effort = None if clamped in (None, "off") else clamped

    return stream(model, context, {**base, "reasoning_effort": reasoning_effort})


class OpenAIResponsesStreams:
    def stream(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return stream(model, context, options)

    def stream_simple(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return stream_simple(model, context, options)


def openai_responses_api() -> OpenAIResponsesStreams:
    return OpenAIResponsesStreams()
