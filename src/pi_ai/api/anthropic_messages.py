from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from pi_ai.api.anthropic_messages_compat import (
    FINE_GRAINED_TOOL_STREAMING_BETA,
    INTERLEAVED_THINKING_BETA,
    force_adaptive_thinking,
    get_anthropic_compat,
)
from pi_ai.api.anthropic_messages_messages import CLAUDE_CODE_VERSION
from pi_ai.api.anthropic_messages_params import (
    build_params,
    map_thinking_level_to_effort,
    resolve_cache_retention,
    should_use_fine_grained_tool_streaming_beta,
)
from pi_ai.api.anthropic_messages_stream import AnthropicStreamParser
from pi_ai.api.github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from pi_ai.api.simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_max_tokens_to_context,
)
from pi_ai.cost import clamp_thinking_level
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.http.client import OpenProviderStream, merge_headers, open_stream_post
from pi_ai.http.errors import ProviderHTTPError, format_provider_error, normalize_provider_error
from pi_ai.http.retry import retry_provider_request
from pi_ai.http.sse import iter_sse
from pi_ai.types import Context, DoneEvent, ErrorEvent, Model, StartEvent

#: SDK 原本自动发送这个头，改成裸 HTTP 后必须显式补上，漏了直接 400。
ANTHROPIC_VERSION = "2023-06-01"

OAUTH_TOKEN_MARKER = "sk-ant-oat"
OAUTH_BETAS = ("claude-code-20250219", "oauth-2025-04-20")

MIN_ANSWER_TOKENS = 1024


def build_request_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/messages"


def is_oauth_token(api_key: str | None) -> bool:
    return bool(api_key and OAUTH_TOKEN_MARKER in api_key)


def _has_header(headers: Mapping[str, str | None] | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected and value is not None and value.strip():
            return True
    return False


def assert_request_auth(
    provider: str,
    api_key: str | None,
    headers: Mapping[str, str | None] | None,
) -> None:
    """确认这次请求有可用的凭据。

    调用方可能已经在 header 里放了 authorization / x-api-key。
    与 TS 不同的是，缺凭据时由调用方转成 error 事件，而不是同步抛给用户。
    """
    if api_key:
        return
    for name in ("authorization", "x-api-key", "cf-aig-authorization"):
        if _has_header(headers, name):
            return
    raise ProviderHTTPError(f"No API key for provider: {provider}")


def build_beta_features(model: Model, context: Context, interleaved_thinking: bool) -> list[str]:
    betas: list[str] = []
    if should_use_fine_grained_tool_streaming_beta(model, context):
        betas.append(FINE_GRAINED_TOOL_STREAMING_BETA)
    # 自适应思考模型内建交错思考，不需要这个 beta 头。
    if interleaved_thinking and not force_adaptive_thinking(model):
        betas.append(INTERLEAVED_THINKING_BETA)
    return betas


def build_request_headers(
    model: Model,
    context: Context,
    api_key: str | None,
    *,
    interleaved_thinking: bool = True,
    options_headers: Mapping[str, str | None] | None = None,
    session_id: str | None = None,
) -> tuple[dict[str, str], bool]:
    """拼出请求头，返回 (headers, 是否走 OAuth 身份)。

    合并顺序照搬 TS：固定头 -> 会话亲和 -> 动态头 -> model.headers -> 调用方 headers。
    """
    compat = get_anthropic_compat(model)
    betas = build_beta_features(model, context, interleaved_thinking)
    oauth = False

    base: dict[str, str | None] = {
        "accept": "application/json",
        "anthropic-dangerous-direct-browser-access": "true",
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    dynamic: dict[str, str | None] = {}
    affinity: dict[str, str | None] = {}
    auth: dict[str, str | None] = {}

    if model.provider == "github-copilot":
        dynamic.update(
            build_copilot_dynamic_headers(context.messages, has_copilot_vision_input(context.messages))
        )
        if betas:
            base["anthropic-beta"] = ",".join(betas)
        if api_key:
            auth["authorization"] = f"Bearer {api_key}"
    elif is_oauth_token(api_key):
        oauth = True
        base["anthropic-beta"] = ",".join([*OAUTH_BETAS, *betas])
        base["user-agent"] = f"claude-cli/{CLAUDE_CODE_VERSION}"
        base["x-app"] = "cli"
        auth["authorization"] = f"Bearer {api_key}"
    else:
        if betas:
            base["anthropic-beta"] = ",".join(betas)
        if session_id and compat.send_session_affinity_headers:
            affinity["x-session-affinity"] = session_id
        if api_key:
            auth["x-api-key"] = api_key

    merged = merge_headers(base, affinity, dynamic, auth, model.headers, options_headers)
    return merged, oauth


def stream(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    """向 Anthropic Messages 端点发起流式请求。

    契约：任何失败都不向调用方抛异常，而是产出 error 事件。
    """
    options = dict(options or {})
    event_stream = AssistantMessageEventStream()
    parser_holder: dict[str, AnthropicStreamParser] = {}

    async def run() -> None:
        parser = AnthropicStreamParser(model, tools=context.tools)
        parser_holder["parser"] = parser
        try:
            api_key = options.get("api_key")
            assert_request_auth(model.provider, api_key, options.get("headers"))

            cache_retention = resolve_cache_retention(options.get("cache_retention"), options.get("env"))
            cache_session_id = None if cache_retention == "none" else options.get("session_id")

            headers, is_oauth = build_request_headers(
                model,
                context,
                api_key,
                interleaved_thinking=options.get("interleaved_thinking", True),
                options_headers=options.get("headers"),
                session_id=cache_session_id,
            )
            parser.is_oauth = is_oauth

            params = build_params(model, context, is_oauth, options)
            on_payload = options.get("on_payload")
            if on_payload is not None:
                replaced = on_payload(params, model)
                if asyncio.iscoroutine(replaced):
                    replaced = await replaced
                if replaced is not None:
                    params = replaced

            url = build_request_url(options.get("base_url") or model.base_url)
            await _consume(event_stream, parser, model, url, headers, params, options)
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
    parser: AnthropicStreamParser,
    model: Model,
    url: str,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
    options: Mapping[str, Any],
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
            if not data:
                continue
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            # Anthropic 用一个 error 事件承载流内错误，收到即中止。
            if payload.get("type") == "error":
                detail = payload.get("error") or {}
                raise ProviderHTTPError(
                    detail.get("message") or "Anthropic stream error", error=payload
                )
            for event in parser.feed(payload):
                event_stream.push(event)

    if parser.saw_message_start and not parser.saw_message_stop:
        raise ProviderHTTPError("Anthropic stream ended without message_stop")

    output = parser.output
    if output.stop_reason == "pending":
        raise ProviderHTTPError("Anthropic stream ended without a stop reason")
    if output.stop_reason == "aborted" or output.stop_reason == "error":  # noqa: PLR1714
        raise ProviderHTTPError(output.error_message or "An unknown error occurred")

    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
    event_stream.end()


def _emit_error(
    event_stream: AssistantMessageEventStream,
    parser: AnthropicStreamParser,
    *,
    aborted: bool,
    error: BaseException | None,
) -> None:
    output = parser.output
    output.stop_reason = "aborted" if aborted else "error"
    # TS 这里只用 error.message；改用统一的归一化，让错误展示质量与
    # openai-completions 一致（HTTP 状态码与响应体都会带出来）。
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
    """统一的简化入口：把 reasoning 级别翻译成 Anthropic 的思考参数。"""
    options = dict(options or {})
    base = build_base_options(model, context, options, options.get("api_key"))

    reasoning = options.get("reasoning")
    reasoning = clamp_thinking_level(model, reasoning) if reasoning else None
    if not reasoning or reasoning == "off":
        return stream(model, context, {**base, "thinking_enabled": False})

    if force_adaptive_thinking(model):
        return stream(
            model,
            context,
            {
                **base,
                "thinking_enabled": True,
                "effort": map_thinking_level_to_effort(model, reasoning),
            },
        )

    # base["max_tokens"] 为 None 表示调用方没设上限，交给 helper 用模型上限。
    adjusted_max_tokens, thinking_budget = adjust_max_tokens_for_thinking(
        base.get("max_tokens"), model.max_tokens, reasoning, options.get("thinking_budgets")
    )
    max_tokens = clamp_max_tokens_to_context(model, context, adjusted_max_tokens)

    return stream(
        model,
        context,
        {
            **base,
            "max_tokens": max_tokens,
            "thinking_enabled": True,
            "thinking_budget_tokens": min(thinking_budget, max(0, max_tokens - MIN_ANSWER_TOKENS)),
        },
    )


class AnthropicMessagesStreams:
    """把本模块包装成 ProviderStreams，供 provider 工厂使用。"""

    def stream(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return stream(model, context, options)

    def stream_simple(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return stream_simple(model, context, options)


def anthropic_messages_api() -> AnthropicMessagesStreams:
    return AnthropicMessagesStreams()
