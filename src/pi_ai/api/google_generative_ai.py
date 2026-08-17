from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from pi_ai.api.google_generative_ai_params import (
    build_params,
    get_google_budget,
    get_thinking_level,
    is_gemini3_flash_model,
    is_gemini3_pro_model,
    is_gemma4_model,
    to_rest_body,
)
from pi_ai.api.google_generative_ai_stream import GoogleStreamParser
from pi_ai.api.simple_options import build_base_options
from pi_ai.cost import clamp_thinking_level
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.http.client import OpenProviderStream, merge_headers, open_stream_post
from pi_ai.http.errors import ProviderHTTPError, format_provider_error, normalize_provider_error
from pi_ai.http.retry import retry_provider_request
from pi_ai.http.sse import iter_sse
from pi_ai.types import Context, DoneEvent, ErrorEvent, Model, StartEvent


def build_request_url(base_url: str, model_id: str) -> str:
    """base_url 已经含版本段（.../v1beta），不要再拼一次版本路径。"""
    return f"{base_url.rstrip('/')}/models/{model_id}:streamGenerateContent?alt=sse"


def build_request_headers(
    model: Model,
    api_key: str,
    options_headers: Mapping[str, str | None] | None = None,
) -> dict[str, str]:
    """Google REST 用 x-goog-api-key，不是 Authorization。"""
    merged = merge_headers(model.headers, options_headers)
    merged.setdefault("x-goog-api-key", api_key)
    merged.setdefault("content-type", "application/json")
    merged.setdefault("accept", "text/event-stream")
    return merged


def stream(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    options = dict(options or {})
    event_stream = AssistantMessageEventStream()

    async def run() -> None:
        parser = GoogleStreamParser(model)
        try:
            api_key = options.get("api_key")
            if not api_key:
                raise ProviderHTTPError(f"No API key for provider: {model.provider}")

            # 第一阶段：SDK 参数形状，供 on_payload 使用（与 TS 一致）
            params = build_params(model, context, options)
            on_payload = options.get("on_payload")
            if on_payload is not None:
                replaced = on_payload(params, model)
                if asyncio.iscoroutine(replaced):
                    replaced = await replaced
                if replaced is not None:
                    params = replaced

            # 第二阶段：转成真正发出去的 REST body
            body = to_rest_body(params)
            if options.get("sampling_params"):
                body.update(options["sampling_params"])

            headers = build_request_headers(model, api_key, options.get("headers"))
            url = build_request_url(options.get("base_url") or model.base_url, model.id)

            await _consume(event_stream, parser, model, url, headers, body, options)
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
    parser: GoogleStreamParser,
    model: Model,
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    options: Mapping[str, Any],
) -> None:
    on_response = options.get("on_response")

    async def connect() -> OpenProviderStream:
        return await open_stream_post(
            url=url,
            headers=headers,
            json_body=body,
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
            if not data:
                continue
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            for event in parser.feed(payload):
                event_stream.push(event)

    for event in parser.finish():
        event_stream.push(event)

    output = parser.output
    if output.stop_reason == "pending":
        raise ProviderHTTPError("Google stream ended without a finish reason")
    if output.stop_reason == "aborted" or output.stop_reason == "error":  # noqa: PLR1714
        raise ProviderHTTPError(
            f"Provider stopped with: {output.raw_stop_reason}"
            if output.raw_stop_reason
            else "An unknown error occurred"
        )

    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
    event_stream.end()


def _emit_error(
    event_stream: AssistantMessageEventStream,
    parser: GoogleStreamParser,
    *,
    aborted: bool,
    error: BaseException | None,
) -> None:
    output = parser.output
    output.stop_reason = "aborted" if aborted else "error"
    output.error_message = (
        "Request was aborted" if error is None else format_provider_error(normalize_provider_error(error))
    )
    event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
    event_stream.end()


def stream_simple(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    """把统一的 reasoning 级别翻译成 Gemini 的思考参数。

    Gemini 3 系列与 Gemma 4 用 thinkingLevel，更早的版本用 thinkingBudget。
    """
    options = dict(options or {})
    base = build_base_options(model, context, options, options.get("api_key"))

    reasoning = options.get("reasoning")
    if not reasoning:
        return stream(model, context, {**base, "thinking": {"enabled": False}})

    clamped = clamp_thinking_level(model, reasoning)
    effort = "high" if clamped == "off" else clamped

    if is_gemini3_pro_model(model) or is_gemini3_flash_model(model) or is_gemma4_model(model):
        return stream(
            model,
            context,
            {**base, "thinking": {"enabled": True, "level": get_thinking_level(effort, model)}},
        )

    return stream(
        model,
        context,
        {
            **base,
            "thinking": {
                "enabled": True,
                "budgetTokens": get_google_budget(model, effort, options.get("thinking_budgets")),
            },
        },
    )


class GoogleGenerativeAIStreams:
    def stream(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return stream(model, context, options)

    def stream_simple(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return stream_simple(model, context, options)


def google_generative_ai_api() -> GoogleGenerativeAIStreams:
    return GoogleGenerativeAIStreams()
