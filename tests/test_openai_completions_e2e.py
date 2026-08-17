"""端到端：从注册厂商到跑完一轮带工具调用的对话。

用 respx 模拟一个 OpenAI 兼容端点，走完整链路（Models -> Provider -> 适配器
-> HTTP -> SSE -> 事件流），不依赖任何真实厂商。
"""

import json

import httpx
import pytest
import respx

from pi_ai import Context, TextContent, Tool, ToolResultMessage, create_models, user_text
from pi_ai.providers import deepseek_provider, groq_provider

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"


def sse(*chunks: dict) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def text_turn(text: str, *, prompt_tokens=100, completion_tokens=20) -> str:
    return sse(
        {"id": "r1", "model": "x", "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"id": "r1", "choices": [{"index": 0, "delta": {"content": text}}]},
        {
            "id": "r1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


def tool_turn(name: str, arguments: str) -> str:
    return sse(
        {
            "id": "r1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": name}}]},
                }
            ],
        },
        {
            "id": "r1",
            "choices": [
                {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]}}
            ],
        },
        {
            "id": "r1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 30},
        },
    )


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    collection = create_models()
    collection.set_provider(groq_provider())
    collection.set_provider(deepseek_provider())
    return collection


@respx.mock
async def test_single_turn_text_conversation(models):
    respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("Hello there")))

    model = models.get_models("groq")[0]
    ctx = Context(system_prompt="Be brief.", messages=[user_text("hi")])

    stream = models.stream_simple(model, ctx)
    deltas = [e.delta async for e in stream if e.type == "text_delta"]
    message = await stream.result()

    assert "".join(deltas) == "Hello there"
    assert message.stop_reason == "stop"
    assert message.usage.input == 100
    assert message.usage.output == 20
    assert message.usage.cost.total >= 0
    assert message.provider == "groq"


@respx.mock
async def test_auth_header_comes_from_the_environment(models):
    route = respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("ok")))

    model = models.get_models("groq")[0]
    await models.complete(model, Context(messages=[user_text("hi")]))

    assert route.calls[0].request.headers["authorization"] == "Bearer sk-groq"


@respx.mock
async def test_full_tool_calling_round_trip(models):
    route = respx.post(GROQ_ENDPOINT)
    route.side_effect = [
        httpx.Response(200, text=tool_turn("get_time", '{"tz":"UTC"}')),
        httpx.Response(200, text=text_turn("It is noon.")),
    ]

    model = models.get_models("groq")[0]
    tools = [
        Tool(
            name="get_time",
            description="Get the current time",
            parameters={"type": "object", "properties": {"tz": {"type": "string"}}},
        )
    ]
    ctx = Context(system_prompt="Be brief.", messages=[user_text("what time is it?")], tools=tools)

    first = await models.complete(model, ctx)
    assert first.stop_reason == "toolUse"
    call = first.content[0]
    assert call.name == "get_time"
    assert call.arguments == {"tz": "UTC"}

    ctx.messages.append(first)
    ctx.messages.append(
        ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            content=[TextContent(text="12:00 UTC")],
            is_error=False,
            timestamp=3,
        )
    )

    second = await models.complete(model, ctx)
    assert second.stop_reason == "stop"
    assert second.content[0].text == "It is noon."

    # 第二轮请求应当把工具调用与工具结果都回放上去
    body = json.loads(route.calls[1].request.content)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert body["messages"][2]["tool_calls"][0]["function"]["name"] == "get_time"
    assert body["messages"][3]["content"] == "12:00 UTC"


@respx.mock
async def test_provider_specific_compat_reaches_the_wire(models):
    """DeepSeek 的推断兼容设置应当体现在真实请求体里。"""
    route = respx.post(DEEPSEEK_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("ok")))

    model = next(m for m in models.get_models("deepseek") if m.reasoning)
    await models.complete_simple(
        model, Context(messages=[user_text("hi")]), reasoning="high", max_tokens=200
    )

    body = json.loads(route.calls[0].request.content)
    # deepseek 用 max_tokens 而不是 max_completion_tokens
    assert "max_tokens" in body
    assert "max_completion_tokens" not in body
    # 且用嵌套的 thinking 对象
    assert body["thinking"] == {"type": "enabled"}
    # 不支持 store
    assert "store" not in body


@respx.mock
async def test_errors_surface_as_messages_not_exceptions(models):
    respx.post(GROQ_ENDPOINT).mock(
        return_value=httpx.Response(500, text='{"error":{"message":"upstream exploded"}}')
    )

    model = models.get_models("groq")[0]
    message = await models.complete(model, Context(messages=[user_text("hi")]))

    assert message.stop_reason == "error"
    assert "upstream exploded" in message.error_message


@respx.mock
async def test_unconfigured_provider_surfaces_as_a_message(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    collection = create_models()
    collection.set_provider(groq_provider())

    model = collection.get_models("groq")[0]
    message = await collection.complete(model, Context(messages=[user_text("hi")]))

    assert message.stop_reason == "error"
    assert "not configured" in message.error_message


@respx.mock
async def test_explicit_api_key_overrides_the_environment(models):
    route = respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("ok")))

    model = models.get_models("groq")[0]
    await models.complete(model, Context(messages=[user_text("hi")]), api_key="sk-explicit")

    assert route.calls[0].request.headers["authorization"] == "Bearer sk-explicit"


@respx.mock
async def test_transform_headers_reaches_the_wire(models):
    route = respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("ok")))

    async def transform(headers):
        return {**headers, "x-request-id": "trace-1"}

    model = models.get_models("groq")[0]
    await models.complete(
        model, Context(messages=[user_text("hi")]), transform_headers=transform
    )

    assert route.calls[0].request.headers["x-request-id"] == "trace-1"


@respx.mock
async def test_context_survives_a_json_round_trip(models):
    respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("resumed")))

    model = models.get_models("groq")[0]
    ctx = Context(system_prompt="Be brief.", messages=[user_text("hi")])

    serialized = ctx.model_dump_json(by_alias=True, exclude_none=True)
    restored = Context.model_validate_json(serialized)

    message = await models.complete(model, restored)
    assert message.content[0].text == "resumed"
