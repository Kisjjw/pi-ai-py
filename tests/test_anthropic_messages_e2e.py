"""端到端：Claude 从注册厂商到跑完一轮带工具调用的对话。

也验证跨 provider 交接——M2 的 OpenAI 兼容模型产出的消息喂给 Claude。
"""

import json

import httpx
import pytest
import respx

from pi_ai import Context, TextContent, Tool, ToolResultMessage, create_models, user_text
from pi_ai.providers import anthropic_provider, minimax_provider

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
MINIMAX_ENDPOINT = "https://api.minimax.io/anthropic/v1/messages"


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


def ev_message_delta(stop_reason="end_turn", **usage) -> dict:
    return {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": usage}


EV_MESSAGE_STOP = {"type": "message_stop"}


def text_turn(text: str, *, input_tokens=100, output_tokens=20) -> str:
    return sse(
        ev_message_start(input_tokens=input_tokens, output_tokens=0),
        ev_block_start(0, {"type": "text", "text": ""}),
        ev_block_delta(0, {"type": "text_delta", "text": text}),
        ev_block_stop(0),
        ev_message_delta(output_tokens=output_tokens),
        EV_MESSAGE_STOP,
    )


def tool_turn(name: str, arguments: str) -> str:
    return sse(
        ev_message_start(input_tokens=150),
        ev_block_start(0, {"type": "tool_use", "id": "toolu_1", "name": name, "input": {}}),
        ev_block_delta(0, {"type": "input_json_delta", "partial_json": arguments}),
        ev_block_stop(0),
        ev_message_delta(stop_reason="tool_use", output_tokens=30),
        EV_MESSAGE_STOP,
    )


def thinking_turn() -> str:
    return sse(
        ev_message_start(input_tokens=10),
        ev_block_start(0, {"type": "thinking", "thinking": ""}),
        ev_block_delta(0, {"type": "thinking_delta", "thinking": "let me think"}),
        ev_block_delta(0, {"type": "signature_delta", "signature": "sig-1"}),
        ev_block_stop(0),
        ev_block_start(1, {"type": "text", "text": ""}),
        ev_block_delta(1, {"type": "text_delta", "text": "4"}),
        ev_block_stop(1),
        ev_message_delta(output_tokens=50, output_tokens_details={"thinking_tokens": 40}),
        EV_MESSAGE_STOP,
    )


@pytest.fixture
def models(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-minimax")
    collection = create_models()
    collection.set_provider(anthropic_provider())
    collection.set_provider(minimax_provider())
    return collection


def claude(models):
    return next(m for m in models.get_models("anthropic") if "sonnet" in m.id)


@respx.mock
async def test_single_turn_text_conversation(models):
    respx.post(ANTHROPIC_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("Hello there")))

    ctx = Context(system_prompt="Be brief.", messages=[user_text("hi")])
    s = models.stream_simple(claude(models), ctx)
    deltas = [e.delta async for e in s if e.type == "text_delta"]
    message = await s.result()

    assert "".join(deltas) == "Hello there"
    assert message.stop_reason == "stop"
    assert message.usage.input == 100
    assert message.usage.output == 20
    assert message.usage.cost.total > 0
    assert message.provider == "anthropic"


@respx.mock
async def test_api_key_goes_into_the_x_api_key_header(models):
    route = respx.post(ANTHROPIC_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("ok")))

    await models.complete(claude(models), Context(messages=[user_text("hi")]))

    request = route.calls[0].request
    assert request.headers["x-api-key"] == "sk-ant-api03-test"
    assert request.headers["anthropic-version"] == "2023-06-01"


@respx.mock
async def test_thinking_round_trip_preserves_the_signature(models):
    route = respx.post(ANTHROPIC_ENDPOINT)
    route.side_effect = [
        httpx.Response(200, text=thinking_turn()),
        httpx.Response(200, text=text_turn("Yes.")),
    ]

    ctx = Context(messages=[user_text("what is 2+2?")])
    first = await models.complete_simple(claude(models), ctx, reasoning="medium")

    assert first.content[0].type == "thinking"
    assert first.content[0].thinking == "let me think"
    assert first.content[0].thinking_signature == "sig-1"
    assert first.usage.reasoning == 40

    ctx.messages.append(first)
    ctx.messages.append(user_text("are you sure?"))
    await models.complete_simple(claude(models), ctx, reasoning="medium")

    # 第二轮必须把带签名的思考块原样回放
    body = json.loads(route.calls[1].request.content)
    replayed = body["messages"][1]["content"][0]
    assert replayed == {"type": "thinking", "thinking": "let me think", "signature": "sig-1"}


@respx.mock
async def test_full_tool_calling_round_trip(models):
    route = respx.post(ANTHROPIC_ENDPOINT)
    route.side_effect = [
        httpx.Response(200, text=tool_turn("get_time", '{"tz":"UTC"}')),
        httpx.Response(200, text=text_turn("It is noon.")),
    ]

    tools = [
        Tool(
            name="get_time",
            description="Get the current time",
            parameters={"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]},
        )
    ]
    ctx = Context(system_prompt="Be brief.", messages=[user_text("what time is it?")], tools=tools)

    first = await models.complete(claude(models), ctx)
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
            timestamp=3,
        )
    )

    second = await models.complete(claude(models), ctx)
    assert second.stop_reason == "stop"
    assert second.content[0].text == "It is noon."

    body = json.loads(route.calls[1].request.content)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["content"][0]["type"] == "tool_use"
    assert body["messages"][2]["content"][0]["type"] == "tool_result"
    assert body["tools"][0]["input_schema"]["required"] == ["tz"]


@respx.mock
async def test_cross_provider_handoff_from_an_openai_model(models, monkeypatch):
    """OpenAI 兼容模型产出的消息喂给 Claude 时，思考块应降级成文本。"""
    from pi_ai.providers import groq_provider

    monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
    models.set_provider(groq_provider())

    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=(
                'data: {"id":"r1","choices":[{"index":0,"delta":{"reasoning_content":"groq thought"}}]}\n\n'
                'data: {"id":"r1","choices":[{"index":0,"delta":{"content":"42"}}]}\n\n'
                'data: {"id":"r1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )
    )
    anthropic_route = respx.post(ANTHROPIC_ENDPOINT).mock(
        return_value=httpx.Response(200, text=text_turn("Confirmed."))
    )

    groq_model = models.get_models("groq")[0]
    ctx = Context(messages=[user_text("what is the answer?")])
    groq_reply = await models.complete(groq_model, ctx)
    assert groq_reply.content[0].type == "thinking"

    ctx.messages.append(groq_reply)
    ctx.messages.append(user_text("is that right?"))
    await models.complete(claude(models), ctx)

    body = json.loads(anthropic_route.calls[0].request.content)
    assistant_blocks = body["messages"][1]["content"]
    # 跨模型时思考块降级成普通文本，不会带着别家的签名发给 Anthropic
    assert all(b["type"] == "text" for b in assistant_blocks)
    assert assistant_blocks[0]["text"] == "groq thought"


@respx.mock
async def test_compatible_provider_uses_its_own_endpoint(models):
    route = respx.post(MINIMAX_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("ok")))

    minimax_model = models.get_models("minimax")[0]
    message = await models.complete(minimax_model, Context(messages=[user_text("hi")]))

    assert message.stop_reason == "stop"
    assert route.calls[0].request.headers["x-api-key"] == "sk-minimax"


@respx.mock
async def test_errors_surface_as_messages_not_exceptions(models):
    respx.post(ANTHROPIC_ENDPOINT).mock(
        return_value=httpx.Response(529, text='{"type":"error","error":{"message":"overloaded"}}')
    )

    message = await models.complete(claude(models), Context(messages=[user_text("hi")]))
    assert message.stop_reason == "error"
    assert "overloaded" in message.error_message


@respx.mock
async def test_unconfigured_provider_surfaces_as_a_message(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    collection = create_models()
    collection.set_provider(anthropic_provider())

    model = next(m for m in collection.get_models("anthropic") if "sonnet" in m.id)
    message = await collection.complete(model, Context(messages=[user_text("hi")]))

    assert message.stop_reason == "error"
    assert "not configured" in message.error_message


@respx.mock
async def test_auth_token_env_reaches_the_wire_as_a_bearer_header(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-token")
    route = respx.post(ANTHROPIC_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("ok")))

    collection = create_models()
    collection.set_provider(anthropic_provider())
    model = next(m for m in collection.get_models("anthropic") if "sonnet" in m.id)

    await collection.complete(model, Context(messages=[user_text("hi")]))

    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer bearer-token"
    assert "x-api-key" not in request.headers
