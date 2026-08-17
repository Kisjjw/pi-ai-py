"""Google 厂商注册与端到端链路。"""

import json

import httpx
import pytest
import respx

from pi_ai import Context, TextContent, Tool, ToolResultMessage, create_models, user_text
from pi_ai.providers import google_provider

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def endpoint(model_id: str) -> str:
    return f"{BASE_URL}/models/{model_id}:streamGenerateContent"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def sse(*chunks: dict) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)


def text_turn(text: str) -> str:
    return sse(
        {"responseId": "r1", "candidates": [{"content": {"parts": [{"text": text}]}}]},
        {
            "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20, "totalTokenCount": 120},
        },
    )


def tool_turn(name: str, args: dict) -> str:
    return sse(
        {
            "candidates": [
                {
                    "content": {"parts": [{"functionCall": {"id": "call_1", "name": name, "args": args}}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 150, "candidatesTokenCount": 30, "totalTokenCount": 180},
        }
    )


def thinking_turn(signature: str) -> str:
    return sse(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "let me think", "thought": True, "thoughtSignature": signature},
                            {"text": "4"},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 40,
                "totalTokenCount": 55,
            },
        }
    )


def test_provider_loads_google_models():
    provider = google_provider()
    models = provider.get_models()
    assert models
    assert all(m.api == "google-generative-ai" for m in models)
    assert all(m.provider == "google" for m in models)


async def test_auth_resolves_from_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
    models = create_models()
    models.set_provider(google_provider())
    resolved = await models.get_auth("google")
    assert resolved is not None
    assert resolved.auth.api_key == "sk-gemini"


async def test_provider_is_unconfigured_without_env():
    models = create_models()
    models.set_provider(google_provider())
    assert await models.get_auth("google") is None


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
    collection = create_models()
    collection.set_provider(google_provider())
    return collection


def gemini(models):
    return next(m for m in models.get_models("google") if "2.5-flash" in m.id)


@respx.mock
async def test_single_turn(models):
    model = gemini(models)
    respx.post(endpoint(model.id)).mock(return_value=httpx.Response(200, text=text_turn("Hello there")))

    s = models.stream_simple(model, Context(system_prompt="Be brief.", messages=[user_text("hi")]))
    deltas = [e.delta async for e in s if e.type == "text_delta"]
    message = await s.result()

    assert "".join(deltas) == "Hello there"
    assert message.stop_reason == "stop"
    assert message.usage.input == 100
    assert message.usage.cost.total > 0


@respx.mock
async def test_api_key_goes_into_the_goog_header(models):
    model = gemini(models)
    route = respx.post(endpoint(model.id)).mock(return_value=httpx.Response(200, text=text_turn("ok")))

    await models.complete(model, Context(messages=[user_text("hi")]))
    assert route.calls[0].request.headers["x-goog-api-key"] == "sk-gemini"


@respx.mock
async def test_thought_signature_round_trip(models):
    """签名必须原样回传，否则会断掉推理链。"""
    import base64

    signature = base64.b64encode(b"thought-signature").decode()
    model = gemini(models)
    route = respx.post(endpoint(model.id))
    route.side_effect = [
        httpx.Response(200, text=thinking_turn(signature)),
        httpx.Response(200, text=text_turn("Yes.")),
    ]

    ctx = Context(messages=[user_text("what is 2+2?")])
    first = await models.complete_simple(model, ctx, reasoning="medium")

    assert first.content[0].type == "thinking"
    assert first.content[0].thinking_signature == signature
    assert first.usage.reasoning == 40

    ctx.messages.append(first)
    ctx.messages.append(user_text("are you sure?"))
    await models.complete_simple(model, ctx, reasoning="medium")

    body = json.loads(route.calls[1].request.content)
    model_turn = next(c for c in body["contents"] if c["role"] == "model")
    thought_part = next(p for p in model_turn["parts"] if p.get("thought"))
    assert thought_part["thoughtSignature"] == signature


@respx.mock
async def test_full_tool_calling_round_trip(models):
    model = gemini(models)
    route = respx.post(endpoint(model.id))
    route.side_effect = [
        httpx.Response(200, text=tool_turn("get_time", {"tz": "UTC"})),
        httpx.Response(200, text=text_turn("It is noon.")),
    ]

    tools = [
        Tool(
            name="get_time",
            description="Get the current time",
            parameters={"type": "object", "properties": {"tz": {"type": "string"}}},
        )
    ]
    ctx = Context(messages=[user_text("what time is it?")], tools=tools)

    first = await models.complete(model, ctx)
    assert first.stop_reason == "toolUse"
    call = first.content[0]
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

    second = await models.complete(model, ctx)
    assert second.content[0].text == "It is noon."

    body = json.loads(route.calls[1].request.content)
    roles = [c["role"] for c in body["contents"]]
    assert roles == ["user", "model", "user"]
    assert body["contents"][1]["parts"][0]["functionCall"]["name"] == "get_time"
    assert body["contents"][2]["parts"][0]["functionResponse"]["response"] == {"output": "12:00 UTC"}
    # gemini-2.5 不需要工具调用 id
    assert "id" not in body["contents"][2]["parts"][0]["functionResponse"]


@respx.mock
async def test_errors_surface_as_messages(models):
    model = gemini(models)
    respx.post(endpoint(model.id)).mock(
        return_value=httpx.Response(429, text='{"error":{"message":"quota exceeded"}}')
    )

    message = await models.complete(model, Context(messages=[user_text("hi")]))
    assert message.stop_reason == "error"
    assert "quota exceeded" in message.error_message


@respx.mock
async def test_unconfigured_provider_surfaces_as_a_message():
    collection = create_models()
    collection.set_provider(google_provider())
    model = next(m for m in collection.get_models("google") if "2.5-flash" in m.id)

    message = await collection.complete(model, Context(messages=[user_text("hi")]))
    assert message.stop_reason == "error"
    assert "not configured" in message.error_message


@respx.mock
async def test_cross_provider_handoff_drops_foreign_signatures(models, monkeypatch):
    """别家模型的思考签名不能发给 Gemini——base64 校验与同模型判定双重拦截。"""
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
    model = gemini(models)
    google_route = respx.post(endpoint(model.id)).mock(
        return_value=httpx.Response(200, text=text_turn("Confirmed."))
    )

    groq_model = models.get_models("groq")[0]
    ctx = Context(messages=[user_text("what is the answer?")])
    groq_reply = await models.complete(groq_model, ctx)
    assert groq_reply.content[0].type == "thinking"

    ctx.messages.append(groq_reply)
    ctx.messages.append(user_text("is that right?"))
    await models.complete(model, ctx)

    body = json.loads(google_route.calls[0].request.content)
    model_turn = next(c for c in body["contents"] if c["role"] == "model")
    # 跨模型时思考块降级成普通文本，且不带签名
    assert all("thought" not in p for p in model_turn["parts"])
    assert all("thoughtSignature" not in p for p in model_turn["parts"])


@respx.mock
async def test_all_five_protocols_share_one_collection(monkeypatch):
    from pi_ai.providers import (
        anthropic_messages_providers,
        openai_completions_providers,
        responses_providers,
    )

    collection = create_models()
    for provider in [
        *openai_completions_providers(),
        *anthropic_messages_providers(),
        *responses_providers(),
        google_provider(),
    ]:
        collection.set_provider(provider)

    assert len(collection.get_providers()) == 28
    apis = {m.api for m in collection.get_models()}
    assert apis == {
        "openai-completions",
        "anthropic-messages",
        "openai-responses",
        "azure-openai-responses",
        "google-generative-ai",
    }
