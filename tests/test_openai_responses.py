import json

import httpx
import pytest
import respx
from conftest import drain, gated_response, read_until

from pi_ai.api.openai_responses import (
    build_params,
    build_request_headers,
    build_request_url,
    get_compat,
    get_service_tier_cost_multiplier,
    resolve_api_key,
    stream,
    stream_simple,
)
from pi_ai.http.errors import ProviderHTTPError
from pi_ai.types import Context, CostBreakdown, Model, ModelCost, Tool, Usage, user_text

ENDPOINT = "https://api.openai.com/v1/responses"


def make_model(
    *,
    provider="openai",
    base_url="https://api.openai.com/v1",
    model_id="gpt-5",
    reasoning=True,
    compat=None,
    level_map=None,
) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-responses",
        provider=provider,
        base_url=base_url,
        reasoning=reasoning,
        thinking_level_map=level_map,
        input=["text"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=0.0),
        context_window=200000,
        max_tokens=32000,
        compat=compat,
    )


CTX = Context(messages=[user_text("hi")])


def sse(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def text_response(text="Hello"):
    return sse(
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": text},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": text}]},
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            },
        },
    )


# --- URL 与鉴权 ---


def test_url_appends_the_responses_path():
    assert build_request_url("https://api.openai.com/v1") == ENDPOINT
    assert build_request_url("https://api.openai.com/v1/") == ENDPOINT


def test_api_key_can_come_from_headers():
    assert resolve_api_key("openai", "sk-1", None) == "sk-1"
    assert resolve_api_key("openai", None, {"Authorization": "Bearer x"}) == "unused"
    with pytest.raises(ProviderHTTPError, match="No API key"):
        resolve_api_key("openai", None, None)


# --- 兼容开关 ---


def test_compat_defaults():
    compat = get_compat(make_model())
    assert compat.supports_developer_role is True
    assert compat.session_affinity_format == "openai"
    assert compat.supports_long_cache_retention is True
    assert compat.supports_strict_mode is False
    assert compat.supports_openai_grammar_tools is False
    assert compat.supports_additional_tools is False
    assert compat.supports_tool_search is False
    assert compat.supports_explicit_prompt_cache_mode is False


def test_openrouter_session_affinity_is_detected():
    assert get_compat(make_model(provider="openrouter")).session_affinity_format == "openrouter"
    assert (
        get_compat(make_model(provider="custom", base_url="https://openrouter.ai/api/v1")).session_affinity_format
        == "openrouter"
    )


def test_explicit_compat_overrides_detection():
    model = make_model(compat={"supportsStrictMode": True, "sessionAffinityFormat": "openai-nosession"})
    compat = get_compat(model)
    assert compat.supports_strict_mode is True
    assert compat.session_affinity_format == "openai-nosession"


# --- 请求头 ---


def test_bearer_auth_is_added():
    headers = build_request_headers(make_model(), CTX, get_compat(make_model()), "sk-1")
    assert headers["authorization"] == "Bearer sk-1"


def test_session_affinity_is_sent_without_any_compat_flag():
    """与 Chat Completions 不同，Responses 只要有 sessionId 就发。"""
    model = make_model()
    headers = build_request_headers(model, CTX, get_compat(model), "sk-1", None, "s1")
    assert headers["session_id"] == "s1"
    assert headers["x-client-request-id"] == "s1"


def test_openrouter_session_affinity_uses_one_header():
    model = make_model(provider="openrouter")
    headers = build_request_headers(model, CTX, get_compat(model), "sk-1", None, "s1")
    assert headers["x-session-id"] == "s1"
    assert "session_id" not in headers


def test_nosession_format_omits_the_session_id_header():
    model = make_model(compat={"sessionAffinityFormat": "openai-nosession"})
    headers = build_request_headers(model, CTX, get_compat(model), "sk-1", None, "s1")
    assert "session_id" not in headers
    assert headers["x-client-request-id"] == "s1"


# --- 请求体 ---


def test_core_fields():
    params = build_params(make_model(), CTX)
    assert params["model"] == "gpt-5"
    assert params["stream"] is True
    assert params["store"] is False
    assert params["input"][0]["role"] == "user"


def test_max_output_tokens_has_a_floor_of_16():
    assert build_params(make_model(), CTX, {"max_tokens": 5})["max_output_tokens"] == 16
    assert build_params(make_model(), CTX, {"max_tokens": 500})["max_output_tokens"] == 500


def test_prompt_cache_key_is_sent_unless_retention_is_none():
    assert build_params(make_model(), CTX, {"session_id": "s1"})["prompt_cache_key"] == "s1"
    assert "prompt_cache_key" not in build_params(
        make_model(), CTX, {"session_id": "s1", "cache_retention": "none"}
    )


def test_long_retention_adds_the_24h_field():
    params = build_params(make_model(), CTX, {"session_id": "s1", "cache_retention": "long"})
    assert params["prompt_cache_retention"] == "24h"


def test_explicit_cache_mode_is_only_sent_when_supported():
    model = make_model(compat={"supportsExplicitPromptCacheMode": True})
    params = build_params(model, CTX, {"cache_retention": "none"})
    assert params["prompt_cache_options"] == {"mode": "explicit"}

    params = build_params(make_model(), CTX, {"cache_retention": "none"})
    assert "prompt_cache_options" not in params


def test_tools_are_converted():
    ctx = Context(messages=[user_text("hi")], tools=[Tool(name="t", description="d", parameters={})])
    assert build_params(make_model(), ctx)["tools"][0]["name"] == "t"


def test_service_tier_is_passed_through():
    assert build_params(make_model(), CTX, {"service_tier": "flex"})["service_tier"] == "flex"


def test_reasoning_with_an_effort_adds_encrypted_include():
    params = build_params(make_model(), CTX, {"reasoning_effort": "high"})
    assert params["reasoning"] == {"effort": "high", "summary": "auto"}
    assert params["include"] == ["reasoning.encrypted_content"]


def test_reasoning_summary_alone_defaults_the_effort_to_medium():
    params = build_params(make_model(), CTX, {"reasoning_summary": "detailed"})
    assert params["reasoning"] == {"effort": "medium", "summary": "detailed"}


def test_level_map_translates_the_effort():
    model = make_model(level_map={"high": "xhigh"})
    assert build_params(model, CTX, {"reasoning_effort": "high"})["reasoning"]["effort"] == "xhigh"


def test_without_an_effort_reasoning_is_turned_off():
    assert build_params(make_model(), CTX)["reasoning"] == {"effort": "none"}


def test_off_mapping_is_used_when_present():
    model = make_model(level_map={"off": "minimal"})
    assert build_params(model, CTX)["reasoning"] == {"effort": "minimal"}


def test_null_off_mapping_sends_no_reasoning():
    model = make_model(level_map={"off": None})
    assert "reasoning" not in build_params(model, CTX)


def test_copilot_never_sends_the_off_reasoning():
    model = make_model(provider="github-copilot")
    assert "reasoning" not in build_params(model, CTX)


def test_xai_always_includes_encrypted_reasoning():
    model = make_model(provider="xai")
    assert build_params(model, CTX)["include"] == ["reasoning.encrypted_content"]


def test_non_reasoning_models_get_no_reasoning_field():
    assert "reasoning" not in build_params(make_model(reasoning=False), CTX, {"reasoning_effort": "high"})


def test_sampling_params_override_named_fields():
    params = build_params(make_model(), CTX, {"temperature": 0.1, "sampling_params": {"temperature": 0.9}})
    assert params["temperature"] == 0.9


# --- service tier 定价 ---


@pytest.mark.parametrize(
    "tier,model_id,expected",
    [
        ("flex", "gpt-5", 0.5),
        ("priority", "gpt-5", 2.0),
        ("priority", "gpt-5.5", 2.5),
        (None, "gpt-5", 1.0),
    ],
)
def test_service_tier_multipliers(tier, model_id, expected):
    assert get_service_tier_cost_multiplier(make_model(model_id=model_id), tier) == expected


def test_service_tier_pricing_scales_every_component():
    from pi_ai.api.openai_responses import apply_service_tier_pricing

    usage = Usage(cost=CostBreakdown(input=1.0, output=2.0, cache_read=0.5, cache_write=0.5))
    apply_service_tier_pricing(usage, "flex", make_model())
    assert usage.cost.input == 0.5
    assert usage.cost.output == 1.0
    assert usage.cost.total == 2.0


# --- 端到端 ---


@respx.mock
async def test_text_stream_completes():
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    events = [e.type async for e in s]
    message = await s.result()

    assert events == ["start", "text_start", "text_delta", "text_end", "done"]
    assert message.content[0].text == "Hello"
    assert message.usage.input == 100
    assert message.usage.cost.total > 0


@respx.mock
async def test_tool_call_stream_reports_tool_use():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_time"},
                },
                {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{}"},
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "get_time",
                        "arguments": "{}",
                    },
                },
                {"type": "response.completed", "response": {"id": "r", "status": "completed", "output": []}},
            ),
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "toolUse"
    assert message.content[0].id == "call_1|fc_1"


@respx.mock
async def test_stream_simple_honors_a_base_url_override():
    """中转站场景：base_url 不在 TS 的 SimpleStreamOptions 里，但 Python 传了必须生效。

    没生效的话请求会打到未 mock 的官方端点，适配器把它变成 error 事件，
    于是 stop_reason 不是 stop——这就是这条断言真正在盯的东西。
    """
    relay = respx.post("https://relay.test/v1/responses").mock(
        return_value=httpx.Response(200, text=text_response())
    )

    message = await stream_simple(
        make_model(), CTX, {"api_key": "sk-1", "base_url": "https://relay.test/v1"}
    ).result()

    assert message.stop_reason == "stop"
    assert relay.call_count == 1


@respx.mock
async def test_events_are_delivered_before_the_upstream_finishes():
    """事件必须随上游边到边推送，而不是整条流攒完再一次性推。"""
    response, body = gated_response(
        sse(
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1"},
            },
            {"type": "response.output_text.delta", "output_index": 0, "delta": "A"},
        ),
        sse(
            {"type": "response.output_text.delta", "output_index": 0, "delta": "B"},
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": "AB"}],
                },
            },
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": [], "usage": {}},
            },
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
async def test_missing_terminal_event_is_an_error():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, text=sse({"type": "response.created", "response": {"id": "r"}}))
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert "terminal response event" in message.error_message


@respx.mock
async def test_http_errors_are_prefixed():
    respx.post(ENDPOINT).mock(return_value=httpx.Response(400, text='{"error":{"message":"bad request"}}'))

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert "OpenAI API error" in message.error_message


async def test_missing_api_key_becomes_an_error_event():
    message = await stream(make_model(), CTX, {}).result()
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


async def test_stream_simple_also_reports_missing_keys_as_events():
    message = await stream_simple(make_model(), CTX, {}).result()
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


@respx.mock
async def test_stream_simple_translates_reasoning():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(make_model(), CTX, {"api_key": "sk-1", "reasoning": "medium"}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["reasoning"]["effort"] == "medium"
    assert body["include"] == ["reasoning.encrypted_content"]
