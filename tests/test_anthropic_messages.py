import json

import httpx
import pytest
import respx
from conftest import drain, gated_response, read_until

from pi_ai.api.anthropic_messages import (
    ANTHROPIC_VERSION,
    assert_request_auth,
    build_beta_features,
    build_request_headers,
    build_request_url,
    is_oauth_token,
    stream,
    stream_simple,
)
from pi_ai.catalog import load_provider_models
from pi_ai.http.errors import ProviderHTTPError
from pi_ai.types import Context, Model, ModelCost, Tool, user_text

ENDPOINT = "https://api.anthropic.com/v1/messages"


def make_model(*, provider="anthropic", model_id="claude-sonnet-4-5", compat=None, headers=None) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="anthropic-messages",
        provider=provider,
        base_url="https://api.anthropic.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        context_window=200000,
        max_tokens=8192,
        compat=compat,
        headers=headers,
    )


CTX = Context(messages=[user_text("hi")])
CTX_WITH_TOOLS = Context(
    messages=[user_text("hi")], tools=[Tool(name="Read", description="d", parameters={"type": "object"})]
)


def sse(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def ev_message_start(**usage) -> dict:
    return {"type": "message_start", "message": {"id": "msg_1", "usage": usage}}


def ev_block_start(index: int, content_block: dict) -> dict:
    return {"type": "content_block_start", "index": index, "content_block": content_block}


def ev_block_delta(index: int, delta: dict) -> dict:
    return {"type": "content_block_delta", "index": index, "delta": delta}


def ev_block_stop(index: int) -> dict:
    return {"type": "content_block_stop", "index": index}


def ev_message_delta(stop_reason="end_turn", usage=None, **delta_extra) -> dict:
    event: dict = {"type": "message_delta", "delta": {"stop_reason": stop_reason, **delta_extra}}
    if usage is not None:
        event["usage"] = usage
    return event


EV_MESSAGE_STOP = {"type": "message_stop"}


def text_response(text="Hello"):
    return sse(
        ev_message_start(input_tokens=100, output_tokens=0),
        ev_block_start(0, {"type": "text", "text": ""}),
        ev_block_delta(0, {"type": "text_delta", "text": text}),
        ev_block_stop(0),
        ev_message_delta(usage={"output_tokens": 20}),
        EV_MESSAGE_STOP,
    )


# --- URL 与鉴权 ---


def test_url_appends_the_messages_path():
    assert build_request_url("https://api.anthropic.com") == ENDPOINT
    assert build_request_url("https://api.anthropic.com/") == ENDPOINT


def test_oauth_tokens_are_detected_by_prefix():
    assert is_oauth_token("sk-ant-oat01-abc") is True
    assert is_oauth_token("sk-ant-api03-abc") is False
    assert is_oauth_token(None) is False


def test_auth_can_come_from_headers():
    assert_request_auth("anthropic", "sk-1", None)
    assert_request_auth("anthropic", None, {"x-api-key": "k"})
    assert_request_auth("anthropic", None, {"Authorization": "Bearer k"})
    assert_request_auth("anthropic", None, {"cf-aig-authorization": "Bearer k"})

    with pytest.raises(ProviderHTTPError, match="No API key"):
        assert_request_auth("anthropic", None, None)


# --- 请求头 ---


def test_api_key_uses_the_x_api_key_header():
    headers, oauth = build_request_headers(make_model(), CTX, "sk-ant-api03-abc")
    assert headers["x-api-key"] == "sk-ant-api03-abc"
    assert "authorization" not in headers
    assert oauth is False


def test_anthropic_version_header_is_always_sent():
    headers, _ = build_request_headers(make_model(), CTX, "sk-1")
    assert headers["anthropic-version"] == ANTHROPIC_VERSION


def test_browser_access_header_is_sent():
    headers, _ = build_request_headers(make_model(), CTX, "sk-1")
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"
    assert headers["accept"] == "application/json"


def test_oauth_uses_bearer_auth_and_claude_code_identity_headers():
    headers, oauth = build_request_headers(make_model(), CTX, "sk-ant-oat01-abc")
    assert oauth is True
    assert headers["authorization"] == "Bearer sk-ant-oat01-abc"
    assert "x-api-key" not in headers
    assert headers["user-agent"].startswith("claude-cli/")
    assert headers["x-app"] == "cli"
    assert "claude-code-20250219" in headers["anthropic-beta"]
    assert "oauth-2025-04-20" in headers["anthropic-beta"]


def test_copilot_uses_bearer_auth():
    headers, oauth = build_request_headers(make_model(provider="github-copilot"), CTX, "gh-token")
    assert oauth is False
    assert headers["authorization"] == "Bearer gh-token"
    assert headers["X-Initiator"] == "user"


def test_interleaved_thinking_beta_is_sent_by_default():
    assert build_beta_features(make_model(), CTX, True) == ["interleaved-thinking-2025-05-14"]


def test_interleaved_thinking_beta_can_be_disabled():
    assert build_beta_features(make_model(), CTX, False) == []


def test_adaptive_thinking_models_skip_the_interleaved_beta():
    model = make_model(compat={"forceAdaptiveThinking": True})
    assert build_beta_features(model, CTX, True) == []


def test_fine_grained_beta_is_added_when_eager_streaming_is_unsupported():
    model = make_model(compat={"supportsEagerToolInputStreaming": False})
    betas = build_beta_features(model, CTX_WITH_TOOLS, True)
    assert "fine-grained-tool-streaming-2025-05-14" in betas


def test_session_affinity_header_requires_the_compat_flag():
    headers, _ = build_request_headers(make_model(), CTX, "sk-1", session_id="s1")
    assert "x-session-affinity" not in headers

    model = make_model(compat={"sendSessionAffinityHeaders": True})
    headers, _ = build_request_headers(model, CTX, "sk-1", session_id="s1")
    assert headers["x-session-affinity"] == "s1"


def test_caller_headers_override_model_headers():
    model = make_model(headers={"x-a": "model", "x-b": "model"})
    headers, _ = build_request_headers(
        model, CTX, "sk-1", options_headers={"x-b": "caller", "x-c": "caller"}
    )
    assert headers["x-a"] == "model"
    assert headers["x-b"] == "caller"
    assert headers["x-c"] == "caller"


# --- 成功路径 ---


@respx.mock
async def test_text_stream_completes():
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    events = [e.type async for e in s]
    message = await s.result()

    assert events == ["start", "text_start", "text_delta", "text_end", "done"]
    assert message.stop_reason == "stop"
    assert message.content[0].text == "Hello"
    assert message.usage.input == 100
    assert message.usage.output == 20
    assert message.usage.cost.total > 0


@respx.mock
async def test_request_body_shape():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    ctx = Context(system_prompt="Be brief.", messages=[user_text("hi")])
    await stream(make_model(), ctx, {"api_key": "sk-1", "max_tokens": 500}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "claude-sonnet-4-5"
    assert body["stream"] is True
    assert body["max_tokens"] == 500
    assert body["system"][0]["text"] == "Be brief."


@respx.mock
async def test_oauth_request_carries_the_identity_block():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream(make_model(), CTX, {"api_key": "sk-ant-oat01-abc"}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["system"][0]["text"].startswith("You are Claude Code")


@respx.mock
async def test_tool_call_stream():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(
                ev_message_start(input_tokens=1),
                ev_block_start(0, {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}}),
                ev_block_delta(0, {"type": "input_json_delta", "partial_json": '{"path":"a.txt"}'}),
                ev_block_stop(0),
                ev_message_delta(stop_reason="tool_use"),
                EV_MESSAGE_STOP,
            ),
        )
    )

    message = await stream(make_model(), CTX_WITH_TOOLS, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "toolUse"
    assert message.content[0].name == "Read"
    assert message.content[0].arguments == {"path": "a.txt"}


@respx.mock
async def test_on_payload_can_replace_the_body():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream(
        make_model(), CTX, {"api_key": "sk-1", "on_payload": lambda p, m: {**p, "custom": "x"}}
    ).result()
    assert json.loads(route.calls[0].request.content)["custom"] == "x"


# --- 错误路径 ---


@respx.mock
async def test_http_error_becomes_an_error_event():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(429, text='{"error":{"message":"rate limited"}}')
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert "rate limited" in message.error_message


async def test_missing_api_key_becomes_an_error_event():
    message = await stream(make_model(), CTX, {}).result()
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


async def test_stream_simple_also_reports_missing_keys_as_events():
    message = await stream_simple(make_model(), CTX, {}).result()
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


@respx.mock
async def test_inline_error_event_aborts_the_stream():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(
                ev_message_start(),
                {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}},
            ),
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert "overloaded" in message.error_message


@respx.mock
async def test_stream_without_a_stop_reason_is_an_error():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(ev_message_start(), EV_MESSAGE_STOP),
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert "without a stop reason" in message.error_message


@respx.mock
async def test_truncated_stream_without_message_stop_is_an_error():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(ev_message_start(), ev_message_delta()),
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert "message_stop" in message.error_message


@respx.mock
async def test_partial_content_survives_an_error():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(
                ev_message_start(),
                ev_block_start(0, {"type": "text", "text": ""}),
                ev_block_delta(0, {"type": "text_delta", "text": "partial"}),
                ev_message_delta(stop_reason="refusal", stop_details={"explanation": "nope"}),
                EV_MESSAGE_STOP,
            ),
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert message.error_message == "nope"
    assert message.content[0].text == "partial"


# --- 流式交付 ---


@respx.mock
async def test_events_are_delivered_before_the_upstream_finishes():
    """事件必须随上游边到边推送，而不是整条流攒完再一次性推。"""
    response, body = gated_response(
        sse(
            ev_message_start(input_tokens=10, output_tokens=0),
            ev_block_start(0, {"type": "text", "text": ""}),
            ev_block_delta(0, {"type": "text_delta", "text": "A"}),
        ),
        sse(
            ev_block_delta(0, {"type": "text_delta", "text": "B"}),
            ev_block_stop(0),
            ev_message_delta(),
            EV_MESSAGE_STOP,
        ),
    )
    respx.post(ENDPOINT).mock(return_value=response)

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    events = s.__aiter__()

    seen = await read_until(events, "text_delta")
    assert [e.type for e in seen] == ["start", "text_start", "text_delta"]
    assert seen[-1].delta == "A"
    assert seen[-1].partial.content[0].text == "A"
    assert body.yielded == 1

    body.open()
    rest = await drain(events)
    assert rest[-1].type == "done"
    assert (await s.result()).content[0].text == "AB"


@respx.mock
async def test_inline_error_events_are_not_retried():
    """流内 error 事件是可重试形状的 ProviderHTTPError，但流已经开始，不能重放。"""
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(
                ev_message_start(),
                {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}},
            ),
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1", "max_retries": 3}).result()

    assert message.stop_reason == "error"
    assert route.call_count == 1


# --- stream_simple ---


@respx.mock
async def test_stream_simple_without_reasoning_disables_thinking():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert json.loads(route.calls[0].request.content)["thinking"] == {"type": "disabled"}


@respx.mock
async def test_stream_simple_uses_budget_thinking_by_default():
    """预算被答案预留量压过一次。

    build_base_options 总会把 max_tokens 填成模型上限（这里 8192），于是
    max_tokens <= 该级别的默认预算（medium 也是 8192），触发"始终给答案
    留 1024"的兜底，最终预算是 8192-1024。TS 走的是同一条路径。
    """
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(make_model(), CTX, {"api_key": "sk-1", "reasoning": "medium"}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["thinking"]["type"] == "enabled"
    assert body["thinking"]["budget_tokens"] == 8192 - 1024
    assert body["thinking"]["display"] == "summarized"


@respx.mock
async def test_stream_simple_budget_is_not_clamped_on_a_large_model():
    """模型上限远大于预算时，预算保持该级别的默认值。"""
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))
    model = make_model()
    model.max_tokens = 64000

    await stream_simple(model, CTX, {"api_key": "sk-1", "reasoning": "low"}).result()

    assert json.loads(route.calls[0].request.content)["thinking"]["budget_tokens"] == 2048


@respx.mock
async def test_stream_simple_uses_adaptive_thinking_when_forced():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))
    model = make_model(compat={"forceAdaptiveThinking": True})

    await stream_simple(model, CTX, {"api_key": "sk-1", "reasoning": "high"}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}


@respx.mock
async def test_stream_simple_always_leaves_room_for_the_answer():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(
        make_model(), CTX, {"api_key": "sk-1", "reasoning": "high", "max_tokens": 2000}
    ).result()

    body = json.loads(route.calls[0].request.content)
    assert body["thinking"]["budget_tokens"] <= body["max_tokens"] - 1024


@pytest.mark.parametrize(
    "model_id,reasoning,thinking_type,effort",
    [
        ("claude-opus-5-5", None, None, None),
        ("claude-sonnet-5-5", None, "between_tools", None),
        ("claude-haiku-5-5", None, "disabled", None),
        ("claude-opus-5-5", "off", "adaptive", "low"),
        ("claude-sonnet-5-5", "off", "between_tools", None),
        ("claude-haiku-5-5", "off", "disabled", None),
        ("claude-opus-5-5", "max", "adaptive", "max"),
        ("claude-sonnet-5-5", "max", "adaptive", "max"),
        ("claude-haiku-5-5", "max", "adaptive", "max"),
        ("claude-haiku-4-5", "off", "disabled", None),
    ],
)
@respx.mock
async def test_catalogued_claude_models_send_supported_thinking_settings(
    model_id: str, reasoning: str | None, thinking_type: str | None, effort: str | None
) -> None:
    response_text = text_response()  # type: ignore[no-untyped-call]  # Existing fixture helper.
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=response_text))
    model = load_provider_models("anthropic")[model_id]

    reply = await stream_simple(model, CTX, {"api_key": "test-key", "reasoning": reasoning}).result()

    assert reply.stop_reason == "stop"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == model_id
    if thinking_type is None:
        assert "thinking" not in body
    elif thinking_type == "adaptive":
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    else:
        assert body["thinking"] == {"type": thinking_type}
    if effort:
        assert body["output_config"] == {"effort": effort}
    else:
        assert "output_config" not in body
