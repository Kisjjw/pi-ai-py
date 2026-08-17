from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from pi_ai.api.constrained_sampling import create_grammar_tool_input_properties
from pi_ai.api.github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from pi_ai.api.openai_completions_compat import ResolvedCompat, get_compat
from pi_ai.api.openai_completions_params import build_params, resolve_cache_retention
from pi_ai.api.openai_completions_stream import CompletionsStreamParser
from pi_ai.api.simple_options import build_base_options
from pi_ai.cost import clamp_thinking_level
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.http.client import OpenProviderStream, merge_headers, open_stream_post
from pi_ai.http.errors import ProviderHTTPError, format_provider_error, normalize_provider_error
from pi_ai.http.retry import retry_provider_request
from pi_ai.http.sse import iter_sse
from pi_ai.types import Context, DoneEvent, ErrorEvent, Model, StartEvent

DONE_SENTINEL = "[DONE]"


def build_request_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


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
    """解析请求用的 API key。

    调用方已经在 header 里放了 authorization 的话就不需要 key。
    与 TS 不同的是，缺 key 时由调用方转成 error 事件，而不是同步抛给用户。
    """
    if api_key:
        return api_key
    if _has_header(headers, "authorization") or _has_header(headers, "cf-aig-authorization"):
        return "unused"
    raise ProviderHTTPError(f"No API key for provider: {provider}")


def build_request_headers(
    model: Model,
    context: Context,
    compat: ResolvedCompat,
    api_key: str,
    options_headers: Mapping[str, str | None] | None = None,
    session_id: str | None = None,
) -> dict[str, str]:
    """按 TS 的顺序拼出请求头：模型默认 -> 动态 -> 会话亲和 -> 调用方覆盖。"""
    headers: dict[str, str | None] = dict(model.headers or {})

    if model.provider == "github-copilot":
        headers.update(
            build_copilot_dynamic_headers(context.messages, has_copilot_vision_input(context.messages))
        )

    if session_id and compat.send_session_affinity_headers:
        if compat.session_affinity_format == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.session_affinity_format == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id
            headers["x-session-affinity"] = session_id

    merged = merge_headers(headers, options_headers)

    # SDK 原本负责的部分：鉴权与内容类型
    if not _has_header(merged, "authorization") and not _has_header(merged, "cf-aig-authorization"):
        merged["authorization"] = f"Bearer {api_key}"
    merged.setdefault("content-type", "application/json")
    merged.setdefault("accept", "text/event-stream")
    return merged


def stream(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    """向 OpenAI 兼容端点发起流式请求。

    契约：任何失败都不向调用方抛异常，而是产出 error 事件。
    """
    options = dict(options or {})
    event_stream = AssistantMessageEventStream()

    async def run() -> None:
        parser = CompletionsStreamParser(model)
        try:
            compat = get_compat(model)
            api_key = resolve_api_key(model.provider, options.get("api_key"), options.get("headers"))
            grammar_properties = create_grammar_tool_input_properties(
                context.tools, compat.supports_openai_grammar_tools
            )
            cache_retention = resolve_cache_retention(options.get("cache_retention"), options.get("env"))
            cache_session_id = None if cache_retention == "none" else options.get("session_id")

            parser.grammar_tool_input_properties = dict(grammar_properties)

            params = build_params(model, context, options, compat, cache_retention, grammar_properties)
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

            await _consume(
                event_stream, parser, model, url, headers, params, options, compat
            )
        except asyncio.CancelledError:
            _emit_error(event_stream, parser, aborted=True, error=None)
            raise
        except BaseException as error:  # noqa: BLE001 - 契约要求不抛出
            _emit_error(event_stream, parser, aborted=False, error=error)

    task = asyncio.ensure_future(run())
    event_stream.bind_producer(task)
    return event_stream


async def _consume(
    event_stream: AssistantMessageEventStream,
    parser: CompletionsStreamParser,
    model: Model,
    url: str,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
    options: Mapping[str, Any],
    compat: ResolvedCompat,
) -> None:
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
            if not data or data == DONE_SENTINEL:
                if data == DONE_SENTINEL:
                    break
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            for event in parser.feed(chunk):
                event_stream.push(event)

    for event in parser.finish():
        event_stream.push(event)

    output = parser.output
    if output.stop_reason == "aborted":
        raise ProviderHTTPError("Request was aborted")
    if not parser.has_finish_reason and not compat.supports_finish_reason:
        parser.infer_missing_stop_reason()
    if output.stop_reason == "error":
        raise ProviderHTTPError(output.error_message or "Provider returned an error stop reason")
    if (compat.supports_finish_reason and not parser.has_finish_reason) or output.stop_reason == "pending":
        raise ProviderHTTPError("Stream ended without finish_reason")

    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
    event_stream.end()


def _emit_error(
    event_stream: AssistantMessageEventStream,
    parser: CompletionsStreamParser,
    *,
    aborted: bool,
    error: BaseException | None,
) -> None:
    output = parser.output
    output.stop_reason = "aborted" if aborted else "error"
    if error is None:
        output.error_message = "Request was aborted"
    else:
        output.error_message = format_provider_error(normalize_provider_error(error))
        # OpenRouter 会在这个字段里给出额外信息；normalize 已经把解析后的 body
        # 拼进了消息，所以只在不重复时追加。
        raw_metadata = _extract_raw_metadata(error)
        if raw_metadata and raw_metadata not in output.error_message:
            output.error_message += f"\n{raw_metadata}"

    event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
    event_stream.end()


def _extract_raw_metadata(error: BaseException) -> str | None:
    payload = getattr(error, "error", None)
    if not isinstance(payload, dict):
        return None
    inner = payload.get("error")
    if not isinstance(inner, dict):
        return None
    metadata = inner.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("raw")
    return str(raw) if raw else None


def stream_simple(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    """统一的简化入口：把 reasoning 级别翻译成本 API 的具体参数。"""
    options = dict(options or {})
    base = build_base_options(model, context, options, options.get("api_key"))

    reasoning = options.get("reasoning")
    clamped = clamp_thinking_level(model, reasoning) if reasoning else None
    reasoning_effort = None if clamped in (None, "off") else clamped

    return stream(
        model,
        context,
        {
            **base,
            "reasoning_effort": reasoning_effort,
            "tool_choice": options.get("tool_choice"),
            "thinking_budgets": options.get("thinking_budgets"),
        },
    )


class OpenAICompletionsStreams:
    """把本模块包装成 ProviderStreams，供 provider 工厂使用。"""

    def stream(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return stream(model, context, options)

    def stream_simple(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return stream_simple(model, context, options)


def openai_completions_api() -> OpenAICompletionsStreams:
    return OpenAICompletionsStreams()
