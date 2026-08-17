import base64
import json

import httpx
import pytest
import respx
from conftest import drain, gated_response, read_until

from pi_ai.api.google_generative_ai import build_request_headers, build_request_url, stream, stream_simple
from pi_ai.api.google_generative_ai_params import (
    build_params,
    get_disabled_thinking_config,
    get_google_budget,
    get_thinking_level,
    is_gemini3_flash_model,
    is_gemini3_pro_model,
    is_gemma4_model,
    to_rest_body,
)
from pi_ai.api.google_generative_ai_stream import GoogleStreamParser
from pi_ai.types import Context, Model, ModelCost, Tool, user_text

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
VALID_SIG = base64.b64encode(b"sig").decode()


def make_model(*, model_id="gemini-2.5-flash", reasoning=True) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="google-generative-ai",
        provider="google",
        base_url=BASE_URL,
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(input=0.3, output=2.5, cache_read=0.075, cache_write=0.0),
        context_window=1000000,
        max_tokens=64000,
    )


CTX = Context(messages=[user_text("hi")])


def endpoint(model_id="gemini-2.5-flash") -> str:
    return f"{BASE_URL}/models/{model_id}:streamGenerateContent"


def sse(*chunks: dict) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)


def text_response(text="Hello"):
    return sse(
        {
            "responseId": "resp_1",
            "candidates": [{"content": {"parts": [{"text": text}]}}],
        },
        {
            "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 20,
                "totalTokenCount": 120,
            },
        },
    )


# --- 模型族判定 ---


@pytest.mark.parametrize(
    "model_id,pro,flash,gemma",
    [
        ("gemini-3-pro", True, False, False),
        ("gemini-3.1-pro", True, False, False),
        ("gemini-3-flash", False, True, False),
        ("gemini-flash-latest", False, True, False),
        ("gemini-flash-lite-latest", False, True, False),
        ("gemma-4", False, False, True),
        ("gemma4", False, False, True),
        ("gemini-2.5-flash", False, False, False),
    ],
)
def test_model_family_detection(model_id, pro, flash, gemma):
    model = make_model(model_id=model_id)
    assert is_gemini3_pro_model(model) is pro
    assert is_gemini3_flash_model(model) is flash
    assert is_gemma4_model(model) is gemma


# --- 思考配置 ---


def test_disabled_thinking_uses_the_lowest_level_on_gemini_3():
    assert get_disabled_thinking_config(make_model(model_id="gemini-3-pro")) == {"thinkingLevel": "LOW"}
    assert get_disabled_thinking_config(make_model(model_id="gemini-3-flash")) == {"thinkingLevel": "MINIMAL"}
    assert get_disabled_thinking_config(make_model(model_id="gemma-4")) == {"thinkingLevel": "MINIMAL"}


def test_disabled_thinking_uses_a_zero_budget_on_gemini_2():
    assert get_disabled_thinking_config(make_model(model_id="gemini-2.5-flash")) == {"thinkingBudget": 0}


@pytest.mark.parametrize(
    "model_id,effort,expected",
    [
        ("gemini-3-pro", "minimal", "LOW"),
        ("gemini-3-pro", "low", "LOW"),
        ("gemini-3-pro", "medium", "HIGH"),
        ("gemini-3-pro", "high", "HIGH"),
        ("gemma-4", "low", "MINIMAL"),
        ("gemma-4", "high", "HIGH"),
        ("gemini-3-flash", "minimal", "MINIMAL"),
        ("gemini-3-flash", "medium", "MEDIUM"),
    ],
)
def test_thinking_level_mapping(model_id, effort, expected):
    assert get_thinking_level(effort, make_model(model_id=model_id)) == expected


@pytest.mark.parametrize(
    "model_id,effort,expected",
    [
        ("gemini-2.5-pro", "minimal", 128),
        ("gemini-2.5-pro", "high", 32768),
        ("gemini-2.5-flash-lite", "minimal", 512),
        ("gemini-2.5-flash-lite", "high", 24576),
        ("gemini-2.5-flash", "minimal", 128),
        ("gemini-2.5-flash", "high", 24576),
        ("gemini-1.5-pro", "high", -1),
    ],
)
def test_budget_tables(model_id, effort, expected):
    assert get_google_budget(make_model(model_id=model_id), effort) == expected


def test_flash_lite_is_matched_before_flash():
    """判定顺序敏感：flash-lite 必须先匹配，否则会被 flash 吞掉。"""
    assert get_google_budget(make_model(model_id="gemini-2.5-flash-lite"), "minimal") == 512


def test_custom_budgets_win():
    assert get_google_budget(make_model(model_id="gemini-2.5-pro"), "low", {"low": 99}) == 99


# --- 两阶段请求体 ---


def test_build_params_returns_the_sdk_shape():
    """on_payload 拿到的必须是 SDK 形状，与 TS 一致。"""
    ctx = Context(system_prompt="be brief", messages=[user_text("hi")])
    params = build_params(make_model(), ctx, {"temperature": 0.5, "max_tokens": 100})

    assert set(params) == {"model", "contents", "config"}
    assert params["model"] == "gemini-2.5-flash"
    assert params["config"]["temperature"] == 0.5
    assert params["config"]["maxOutputTokens"] == 100
    assert params["config"]["systemInstruction"] == "be brief"


def test_rest_body_moves_sampling_fields_into_generation_config():
    params = build_params(make_model(), CTX, {"temperature": 0.5, "max_tokens": 100})
    body = to_rest_body(params)

    assert "model" not in body
    assert body["generationConfig"] == {"temperature": 0.5, "maxOutputTokens": 100}


def test_rest_body_wraps_the_system_instruction():
    ctx = Context(system_prompt="be brief", messages=[user_text("hi")])
    body = to_rest_body(build_params(make_model(), ctx))
    assert body["systemInstruction"] == {"parts": [{"text": "be brief"}]}


def test_rest_body_keeps_tools_at_the_top_level():
    ctx = Context(messages=[user_text("hi")], tools=[Tool(name="t", description="d", parameters={})])
    body = to_rest_body(build_params(make_model(), ctx))
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "t"


def test_thinking_config_lands_in_generation_config():
    params = build_params(make_model(), CTX, {"thinking": {"enabled": True, "budgetTokens": 2048}})
    body = to_rest_body(params)
    assert body["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 2048,
    }


def test_thinking_level_takes_precedence_over_budget():
    params = build_params(
        make_model(), CTX, {"thinking": {"enabled": True, "level": "HIGH", "budgetTokens": 2048}}
    )
    assert params["config"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "HIGH"}


def test_disabled_thinking_is_applied():
    params = build_params(make_model(), CTX, {"thinking": {"enabled": False}})
    assert params["config"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_non_reasoning_models_get_no_thinking_config():
    params = build_params(make_model(reasoning=False), CTX, {"thinking": {"enabled": True}})
    assert "thinkingConfig" not in params["config"]


def test_tool_config_is_added_for_strict_tools():
    tool = Tool(
        name="t",
        description="d",
        parameters={"type": "object"},
        constrained_sampling={"type": "json_schema", "strict": "prefer"},
    )
    ctx = Context(messages=[user_text("hi")], tools=[tool])
    params = build_params(make_model(model_id="gemini-3-pro"), ctx)
    assert params["config"]["toolConfig"] == {"functionCallingConfig": {"mode": "VALIDATED"}}


# --- URL 与请求头 ---


def test_url_includes_the_model_and_sse_marker():
    assert build_request_url(BASE_URL, "gemini-2.5-flash") == (
        f"{BASE_URL}/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )


def test_url_does_not_double_the_version_path():
    """base_url 已经含 /v1beta，不能再拼一次。"""
    url = build_request_url(BASE_URL + "/", "m")
    assert url.count("/v1beta") == 1


def test_google_uses_the_api_key_header():
    headers = build_request_headers(make_model(), "sk-gemini")
    assert headers["x-goog-api-key"] == "sk-gemini"
    assert "authorization" not in headers


def test_caller_headers_win():
    model = make_model()
    model.headers = {"x-a": "model"}
    headers = build_request_headers(model, "sk-1", {"x-a": "caller"})
    assert headers["x-a"] == "caller"


# --- 流式解析 ---


def run(parser, chunks):
    out = []
    for chunk in chunks:
        out.extend(parser.feed(chunk))
    out.extend(parser.finish())
    return out


def types_of(events):
    return [e.type for e in events]


def test_text_chunks_merge_into_one_block():
    parser = GoogleStreamParser(make_model())
    events = run(
        parser,
        [
            {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]},
        ],
    )
    assert types_of(events) == ["text_start", "text_delta", "text_delta", "text_end"]
    assert parser.output.content[0].text == "Hello"


def test_switching_between_thinking_and_text_opens_a_new_block():
    parser = GoogleStreamParser(make_model())
    events = run(
        parser,
        [
            {"candidates": [{"content": {"parts": [{"text": "think", "thought": True}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "answer"}]}}]},
        ],
    )
    assert types_of(events) == [
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
    ]
    assert parser.output.content[0].thinking == "think"
    assert parser.output.content[1].text == "answer"


def test_thought_signatures_are_retained_across_deltas():
    parser = GoogleStreamParser(make_model())
    run(
        parser,
        [
            {"candidates": [{"content": {"parts": [{"text": "a", "thoughtSignature": VALID_SIG}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "b"}]}}]},
        ],
    )
    assert parser.output.content[0].text_signature == VALID_SIG


def test_function_calls_emit_three_events_at_once():
    parser = GoogleStreamParser(make_model())
    events = run(
        parser,
        [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"functionCall": {"id": "c1", "name": "get_time", "args": {"tz": "UTC"}}}]
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        ],
    )
    assert types_of(events) == ["toolcall_start", "toolcall_delta", "toolcall_end"]
    call = parser.output.content[0]
    assert call.id == "c1"
    assert call.arguments == {"tz": "UTC"}
    assert parser.output.stop_reason == "toolUse"


def test_missing_tool_call_ids_are_generated():
    parser = GoogleStreamParser(make_model())
    run(parser, [{"candidates": [{"content": {"parts": [{"functionCall": {"name": "t", "args": {}}}]}}]}])
    assert parser.output.content[0].id.startswith("t_")


def test_duplicate_tool_call_ids_are_regenerated():
    parser = GoogleStreamParser(make_model())
    run(
        parser,
        [
            {"candidates": [{"content": {"parts": [{"functionCall": {"id": "dup", "name": "a", "args": {}}}]}}]},
            {"candidates": [{"content": {"parts": [{"functionCall": {"id": "dup", "name": "b", "args": {}}}]}}]},
        ],
    )
    ids = [b.id for b in parser.output.content]
    assert ids[0] == "dup"
    assert ids[1] != "dup"


def test_a_function_call_closes_the_open_text_block():
    parser = GoogleStreamParser(make_model())
    events = run(
        parser,
        [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "calling"}, {"functionCall": {"id": "c1", "name": "t", "args": {}}}]
                        }
                    }
                ]
            }
        ],
    )
    assert types_of(events) == [
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
    ]


def test_usage_adds_thoughts_to_the_output_count():
    """与其它厂商不同：Google 的 candidatesTokenCount 不含思考 token。"""
    parser = GoogleStreamParser(make_model())
    run(
        parser,
        [
            {
                "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 1000,
                    "cachedContentTokenCount": 200,
                    "candidatesTokenCount": 50,
                    "thoughtsTokenCount": 30,
                    "totalTokenCount": 1080,
                },
            }
        ],
    )
    usage = parser.output.usage
    assert usage.input == 800
    assert usage.cache_read == 200
    assert usage.output == 80
    assert usage.reasoning == 30
    assert usage.cost.total > 0


def test_response_id_is_kept_from_the_first_chunk():
    parser = GoogleStreamParser(make_model())
    run(
        parser,
        [
            {"responseId": "first", "candidates": [{"content": {"parts": [{"text": "a"}]}}]},
            {"responseId": "second", "candidates": [{"content": {"parts": [{"text": "b"}]}}]},
        ],
    )
    assert parser.output.response_id == "first"


# --- 端到端 ---


@respx.mock
async def test_text_stream_completes():
    respx.post(endpoint()).mock(return_value=httpx.Response(200, text=text_response()))

    s = stream(make_model(), CTX, {"api_key": "sk-1"})
    events = [e.type async for e in s]
    message = await s.result()

    assert events == ["start", "text_start", "text_delta", "text_end", "done"]
    assert message.content[0].text == "Hello"
    assert message.usage.input == 100
    assert message.usage.cost.total > 0


@respx.mock
async def test_request_body_is_the_rest_shape():
    route = respx.post(endpoint()).mock(return_value=httpx.Response(200, text=text_response()))

    ctx = Context(system_prompt="be brief", messages=[user_text("hi")])
    await stream(make_model(), ctx, {"api_key": "sk-1", "max_tokens": 100}).result()

    body = json.loads(route.calls[0].request.content)
    assert "model" not in body
    assert body["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert body["generationConfig"]["maxOutputTokens"] == 100
    assert route.calls[0].request.headers["x-goog-api-key"] == "sk-1"


@respx.mock
async def test_on_payload_sees_the_sdk_shape_and_can_replace_it():
    route = respx.post(endpoint()).mock(return_value=httpx.Response(200, text=text_response()))
    seen = {}

    def on_payload(payload, model):
        seen.update(payload)
        return {**payload, "config": {**payload["config"], "temperature": 0.9}}

    await stream(make_model(), CTX, {"api_key": "sk-1", "on_payload": on_payload}).result()

    # 回调看到的是 SDK 形状
    assert set(seen) == {"model", "contents", "config"}
    # 线上发出去的是 REST 形状，且替换生效
    body = json.loads(route.calls[0].request.content)
    assert body["generationConfig"]["temperature"] == 0.9


@respx.mock
async def test_events_are_delivered_before_the_upstream_finishes():
    """事件必须随上游边到边推送，而不是整条流攒完再一次性推。"""
    response, body = gated_response(
        sse({"responseId": "resp_1", "candidates": [{"content": {"parts": [{"text": "A"}]}}]}),
        sse(
            {"candidates": [{"content": {"parts": [{"text": "B"}]}}]},
            {
                "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 12,
                },
            },
        ),
    )
    respx.post(endpoint()).mock(return_value=response)

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
async def test_error_finish_reasons_surface_the_raw_reason():
    respx.post(endpoint()).mock(
        return_value=httpx.Response(
            200, text=sse({"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]})
        )
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert "SAFETY" in message.error_message


@respx.mock
async def test_stream_without_a_finish_reason_is_an_error():
    respx.post(endpoint()).mock(
        return_value=httpx.Response(200, text=sse({"candidates": [{"content": {"parts": [{"text": "a"}]}}]}))
    )

    message = await stream(make_model(), CTX, {"api_key": "sk-1"}).result()
    assert message.stop_reason == "error"
    assert "without a finish reason" in message.error_message


async def test_missing_api_key_becomes_an_error_event():
    message = await stream(make_model(), CTX, {}).result()
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


async def test_stream_simple_also_reports_missing_keys_as_events():
    message = await stream_simple(make_model(), CTX, {}).result()
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


@respx.mock
async def test_stream_simple_uses_a_budget_for_gemini_2():
    route = respx.post(endpoint()).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(make_model(), CTX, {"api_key": "sk-1", "reasoning": "medium"}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 8192}


@respx.mock
async def test_stream_simple_uses_a_level_for_gemini_3():
    model = make_model(model_id="gemini-3-pro")
    route = respx.post(endpoint("gemini-3-pro")).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(model, CTX, {"api_key": "sk-1", "reasoning": "medium"}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "HIGH"}


@respx.mock
async def test_stream_simple_without_reasoning_disables_thinking():
    route = respx.post(endpoint()).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(make_model(), CTX, {"api_key": "sk-1"}).result()

    body = json.loads(route.calls[0].request.content)
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
