import asyncio
import json
from dataclasses import replace

import httpx
import pytest
import respx
from conftest import drain, gated_response, read_until

from pi_ai.api.openai_completions import (
    build_request_headers,
    build_request_url,
    resolve_api_key,
    stream,
    stream_simple,
)
from pi_ai.api.openai_completions_compat import detect_compat
from pi_ai.http.errors import ProviderHTTPError
from pi_ai.types import Context, Model, ModelCost, Tool, user_text

ENDPOINT = "https://acme.test/v1/chat/completions"


def make_model(*, provider="acme", base_url="https://acme.test/v1", reasoning=False, headers=None) -> Model:
    return Model(
        id="m",
        name="M",
        api="openai-completions",
        provider=provider,
        base_url=base_url,
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.0, cache_write=0.0),
        context_window=100000,
        max_tokens=4096,
        headers=headers,
    )


def sse(*chunks: dict) -> str:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return body + "data: [DONE]\n\n"


def text_response(text="Hello"):
    return sse(
        {"id": "r1", "choices": [{"index": 0, "delta": {"content": text}}]},
        {
            "id": "r1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


CTX = Context(messages=[user_text("hi")])


# --- URL 与鉴权 ---


def test_url_is_appended_to_the_base_url():
    assert build_request_url("https://acme.test/v1") == ENDPOINT
    assert build_request_url("https://acme.test/v1/") == ENDPOINT


def test_api_key_is_required_unless_a_header_supplies_it():
    assert resolve_api_key("acme", "sk-1", None) == "sk-1"
    assert resolve_api_key("acme", None, {"Authorization": "Bearer x"}) == "unused"
    assert resolve_api_key("acme", None, {"cf-aig-authorization": "Bearer x"}) == "unused"

    with pytest.raises(ProviderHTTPError, match="No API key"):
        resolve_api_key("acme", None, None)
    with pytest.raises(ProviderHTTPError):
        resolve_api_key("acme", None, {"authorization": "   "})


# --- 请求头 ---


def test_bearer_auth_is_added():
    headers = build_request_headers(make_model(), CTX, detect_compat(make_model()), "sk-1")
    assert headers["authorization"] == "Bearer sk-1"
    assert headers["content-type"] == "application/json"


def test_caller_headers_override_model_headers():
    model = make_model(headers={"x-a": "model", "x-b": "model"})
    headers = build_request_headers(
        model, CTX, detect_compat(model), "sk-1", {"x-b": "caller", "x-c": "caller"}
    )
    assert headers["x-a"] == "model"
    assert headers["x-b"] == "caller"
    assert headers["x-c"] == "caller"


def test_existing_authorization_header_is_not_overwritten():
    headers = build_request_headers(
        make_model(), CTX, detect_compat(make_model()), "unused", {"authorization": "Bearer caller"}
    )
    assert headers["authorization"] == "Bearer caller"


def test_openai_session_affinity_headers():
    compat = replace(detect_compat(make_model()), send_session_affinity_headers=True, session_affinity_format="openai")
    headers = build_request_headers(make_model(), CTX, compat, "sk-1", None, "s1")
    assert headers["session_id"] == "s1"
    assert headers["x-client-request-id"] == "s1"
    assert headers["x-session-affinity"] == "s1"


def test_openai_nosession_affinity_headers_omit_session_id():
    compat = replace(
        detect_compat(make_model()), send_session_affinity_headers=True, session_affinity_format="openai-nosession"
    )
    headers = build_request_headers(make_model(), CTX, compat, "sk-1", None, "s1")
    assert "session_id" not in headers
    assert headers["x-client-request-id"] == "s1"


def test_openrouter_session_affinity_uses_a_single_header():
    compat = replace(
        detect_compat(make_model()), send_session_affinity_headers=True, session_affinity_format="openrouter"
    )
    headers = build_request_headers(make_model(), CTX, compat, "sk-1", None, "s1")
    assert headers["x-session-id"] == "s1"
    assert "x-session-affinity" not in headers


def test_session_affinity_headers_are_skipped_when_disabled():
    headers = build_request_headers(make_model(), CTX, detect_compat(make_model()), "sk-1", None, "s1")
    assert "x-session-affinity" not in headers


def test_copilot_dynamic_headers_are_added():
    model = make_model(provider="github-copilot")
    headers = build_request_headers(model, CTX, detect_compat(model), "sk-1")
    assert headers["X-Initiator"] == "user"
    assert headers["Openai-Intent"] == "conversation-edits"


# --- 成功路径 ---


@respx.mock
async def test_simple_text_stream_completes():
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    events = [e.type async for e in s]
    message = await s.result()

    assert events[0] == "start"
    assert events[-1] == "done"
    assert message.stop_reason == "stop"
    assert message.content[0].text == "Hello"
    assert message.usage.input == 10
    assert message.usage.cost.total > 0


@respx.mock
async def test_request_body_is_sent_as_json():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream(make_model(), CTX, {"api_key": "sk-1", "max_tokens": 100}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "m"
    assert body["stream"] is True
    assert body["max_completion_tokens"] == 100
    assert body["messages"] == [{"role": "user", "content": "hi"}]


@respx.mock
async def test_tool_call_stream_reports_tool_use():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(
                {
                    "id": "r1",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "id": "c1", "function": {"name": "get_time", "arguments": "{}"}}
                                ]
                            },
                        }
                    ],
                },
                {"id": "r1", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ),
        )
    )

    ctx = Context(messages=[user_text("hi")], tools=[Tool(name="get_time", description="d", parameters={})])
    message = await stream(make_model(), ctx, {"api_key": "sk-1"}).result()

    assert message.stop_reason == "toolUse"
    assert message.content[0].name == "get_time"


@respx.mock
async def test_on_payload_can_replace_the_request_body():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    def on_payload(payload, model):
        return {**payload, "custom": "injected"}

    await stream(make_model(), CTX, {"api_key": "sk-1", "on_payload": on_payload}).result()
    assert json.loads(route.calls[0].request.content)["custom"] == "injected"


@respx.mock
async def test_on_response_receives_status_and_headers():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, text=text_response(), headers={"x-req": "abc"})
    )
    seen = {}

    def on_response(response, model):
        seen["status"] = response.status
        seen["header"] = response.headers.get("x-req")

    await stream(make_model(), CTX, {"api_key": "sk-1", "on_response": on_response}).result()
    assert seen == {"status": 200, "header": "abc"}


# --- 错误路径 ---


@respx.mock
async def test_http_error_becomes_an_error_event():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(429, text='{"error":{"message":"slow down"}}')
    )

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    events = [e.type async for e in s]
    message = await s.result()

    assert events == ["error"]
    assert message.stop_reason == "error"
    assert "slow down" in message.error_message


async def test_missing_api_key_becomes_an_error_event_instead_of_raising():
    s = stream(make_model(), CTX, {})
    message = await s.result()
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


async def test_stream_simple_also_reports_missing_keys_as_events():
    s = stream_simple(make_model(), CTX, {})
    message = await s.result()
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


@respx.mock
async def test_stream_without_finish_reason_is_an_error():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, text=sse({"id": "r1", "choices": [{"index": 0, "delta": {"content": "a"}}]}))
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert "finish_reason" in message.error_message


@respx.mock
async def test_missing_finish_reason_is_inferred_when_unsupported():
    model = make_model()
    model.compat = {"supportsFinishReason": False}
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, text=sse({"id": "r1", "choices": [{"index": 0, "delta": {"content": "a"}}]}))
    )

    message = await stream(model, CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "stop"
    assert message.content[0].text == "a"


@respx.mock
async def test_content_filter_finish_reason_surfaces_as_an_error():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200, text=sse({"id": "r1", "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]})
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert "content_filter" in message.error_message


@respx.mock
async def test_partial_content_survives_an_error():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(
                {"id": "r1", "choices": [{"index": 0, "delta": {"content": "partial"}}]},
                {"id": "r1", "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]},
            ),
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert message.content[0].text == "partial"


@respx.mock
async def test_openrouter_raw_metadata_is_appended():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            400, text=json.dumps({"error": {"message": "bad", "metadata": {"raw": "upstream said no"}}})
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert "upstream said no" in message.error_message


@respx.mock
async def test_transport_errors_become_error_events():
    respx.post(ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"


# --- stream_simple ---


@respx.mock
async def test_stream_simple_translates_reasoning_to_effort():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(make_model(reasoning=True), CTX, {"api_key": "sk-1", "reasoning": "medium"}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["reasoning_effort"] == "medium"


@respx.mock
async def test_stream_simple_without_reasoning_sends_no_effort():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(make_model(reasoning=True), CTX, {"api_key": "sk-1"}).result()

    assert "reasoning_effort" not in json.loads(route.calls[0].request.content)


@respx.mock
async def test_stream_simple_clamps_max_tokens_to_the_model_cap():
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(make_model(), CTX, {"api_key": "sk-1"}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["max_completion_tokens"] <= 4096


# --- 流式交付 ---


def chunk(**delta) -> str:
    return f'data: {json.dumps({"id": "r1", "choices": [{"index": 0, "delta": delta}]})}\n\n'


STOP_CHUNK = (
    f'data: {json.dumps({"id": "r1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n'
    "data: [DONE]\n\n"
)


def gated_text_response():
    """把一次文本回复切成两段，中间夹一个闸门。"""
    return gated_response(chunk(content="A"), chunk(content="B") + STOP_CHUNK)


@respx.mock
async def test_events_are_delivered_before_the_upstream_finishes():
    """事件必须随上游边到边推送，而不是整条流攒完再一次性推。"""
    response, body = gated_text_response()
    respx.post(ENDPOINT).mock(return_value=response)

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    events = s.__aiter__()

    # 上游还卡在闸门上，第一段对应的事件就应该已经能读到
    seen = await read_until(events, "text_delta")
    assert [e.type for e in seen] == ["start", "text_start", "text_delta"]
    assert seen[-1].delta == "A"
    assert body.yielded == 1

    body.open()
    rest = await drain(events)
    assert rest[-1].type == "done"
    assert (await s.result()).content[0].text == "AB"


@respx.mock
async def test_partial_reflects_the_state_at_each_event():
    """partial 是事件产生那一刻的快照，不是解析结束后的终态。"""
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(
                {"id": "r1", "choices": [{"index": 0, "delta": {"content": "Hel"}}]},
                {"id": "r1", "choices": [{"index": 0, "delta": {"content": "lo"}}]},
                {"id": "r1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ),
        )
    )

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    deltas = [e async for e in s if e.type == "text_delta"]

    assert [e.delta for e in deltas] == ["Hel", "lo"]
    assert [e.partial.content[0].text for e in deltas] == ["Hel", "Hello"]


@respx.mock
async def test_partial_snapshots_do_not_share_mutable_state():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(
                {"id": "r1", "choices": [{"index": 0, "delta": {"content": "a"}}]},
                {"id": "r1", "choices": [{"index": 0, "delta": {"content": "b"}}]},
                {"id": "r1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ),
        )
    )

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    deltas = [e async for e in s if e.type == "text_delta"]
    final = await s.result()

    assert len({id(e.partial) for e in deltas}) == len(deltas)
    assert len({id(e.partial.content[0]) for e in deltas}) == len(deltas)
    # 快照必须与最终消息脱钩，否则后续修改会倒灌进已交付的事件
    assert all(e.partial is not final for e in deltas)
    assert all(e.partial.usage is not final.usage for e in deltas)


@respx.mock
async def test_partial_snapshots_keep_tool_call_arguments_at_event_time():
    respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            text=sse(
                {
                    "id": "r1",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "id": "c1", "function": {"name": "w", "arguments": '{"a":'}}
                                ]
                            },
                        }
                    ],
                },
                {
                    "id": "r1",
                    "choices": [
                        {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}}
                    ],
                },
                {"id": "r1", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ),
        )
    )

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    deltas = [e async for e in s if e.type == "toolcall_delta"]

    assert [e.partial.content[0].arguments for e in deltas] == [{}, {"a": 1}]


@respx.mock
async def test_cancel_settles_the_result_as_aborted():
    response, _body = gated_text_response()
    respx.post(ENDPOINT).mock(return_value=response)

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    await read_until(s.__aiter__(), "text_delta")

    s.cancel()
    message = await s.result()

    assert s.cancelled is True
    assert message.stop_reason == "aborted"
    # 已经收到的增量不会丢
    assert message.content[0].text == "A"


@respx.mock
async def test_cancel_stops_reading_the_upstream_body():
    """cancel() 必须真正中断底层请求，而不只是停止向消费者交付。"""
    response, body = gated_text_response()
    respx.post(ENDPOINT).mock(return_value=response)

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    await read_until(s.__aiter__(), "text_delta")

    s.cancel()
    await s.result()  # 等 aborted 落地，此时取消已经生效

    body.open()
    await asyncio.sleep(0.05)
    assert body.yielded == 1  # 第二段始终没有被读走


@respx.mock
async def test_connect_failures_are_still_retried():
    route = respx.post(ENDPOINT).mock(
        side_effect=[
            httpx.Response(500, text="boom", headers={"retry-after-ms": "0"}),
            httpx.Response(200, text=text_response()),
        ]
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1", "max_retries": 2}).result()

    assert message.stop_reason == "stop"
    assert route.call_count == 2


@respx.mock
async def test_mid_stream_failures_are_not_replayed():
    """响应体一旦开始返回就不能重放，否则会把已经交付的增量再发一遍。"""

    class DroppingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield chunk(content="a").encode()
            raise httpx.ReadError("connection dropped")

    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=DroppingStream()
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1", "max_retries": 3}).result()

    assert message.stop_reason == "error"
    assert route.call_count == 1
