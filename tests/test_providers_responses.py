"""Responses 协议两家厂商，以及端到端链路。"""

import json

import httpx
import pytest
import respx

from pi_ai import Context, TextContent, Tool, ToolResultMessage, create_models, user_text
from pi_ai.providers import azure_openai_responses_provider, openai_provider, responses_providers

OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"
AZURE_ENDPOINT = "https://my-resource.openai.azure.com/openai/v1/responses"

ENV_VARS = (
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_BASE_URL",
    "AZURE_OPENAI_RESOURCE_NAME",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def sse(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def text_turn(text: str, *, input_tokens=100, output_tokens=20) -> str:
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
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            },
        },
    )


def tool_turn(name: str, arguments: str) -> str:
    return sse(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": name},
        },
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": arguments},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": name,
                "arguments": arguments,
            },
        },
        {"type": "response.completed", "response": {"id": "r", "status": "completed", "output": []}},
    )


def reasoning_turn() -> str:
    return sse(
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}},
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "thinking hard"},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"text": "thinking hard"}],
                "encrypted_content": "blob",
            },
        },
        {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1"}},
        {"type": "response.output_text.delta", "output_index": 1, "delta": "4"},
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": "4"}]},
        },
        {"type": "response.completed", "response": {"id": "r", "status": "completed", "output": []}},
    )


# --- 厂商注册 ---


def test_two_responses_providers_are_available():
    assert [p.id for p in responses_providers()] == ["openai", "azure-openai-responses"]


def test_openai_provider_loads_responses_models():
    provider = openai_provider()
    models = provider.get_models()
    assert models
    assert all(m.api == "openai-responses" for m in models)


def test_azure_provider_loads_models_with_empty_base_urls():
    provider = azure_openai_responses_provider()
    models = provider.get_models()
    assert models
    assert all(m.api == "azure-openai-responses" for m in models)
    # Azure 的端点是运行时拼出来的，目录里就是空的
    assert all(m.base_url == "" for m in models)


async def test_openai_auth_resolves(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    models = create_models()
    models.set_provider(openai_provider())
    resolved = await models.get_auth("openai")
    assert resolved is not None
    assert resolved.auth.api_key == "sk-openai"


async def test_azure_auth_resolves(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-azure")
    models = create_models()
    models.set_provider(azure_openai_responses_provider())
    resolved = await models.get_auth("azure-openai-responses")
    assert resolved is not None
    assert resolved.auth.api_key == "sk-azure"


async def test_providers_are_unconfigured_without_env():
    models = create_models()
    for provider in responses_providers():
        models.set_provider(provider)
    assert await models.get_auth("openai") is None
    assert await models.get_auth("azure-openai-responses") is None


# --- 端到端 ---


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-azure")
    monkeypatch.setenv("AZURE_OPENAI_RESOURCE_NAME", "my-resource")
    collection = create_models()
    for provider in responses_providers():
        collection.set_provider(provider)
    return collection


def gpt(models):
    return next(m for m in models.get_models("openai") if m.reasoning)


@respx.mock
async def test_openai_single_turn(models):
    respx.post(OPENAI_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("Hello there")))

    ctx = Context(system_prompt="Be brief.", messages=[user_text("hi")])
    s = models.stream_simple(gpt(models), ctx)
    deltas = [e.delta async for e in s if e.type == "text_delta"]
    message = await s.result()

    assert "".join(deltas) == "Hello there"
    assert message.stop_reason == "stop"
    assert message.usage.input == 100
    assert message.usage.cost.total > 0


@respx.mock
async def test_openai_sends_a_developer_role_system_prompt(models):
    route = respx.post(OPENAI_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("ok")))

    ctx = Context(system_prompt="Be brief.", messages=[user_text("hi")])
    await models.complete(gpt(models), ctx)

    body = json.loads(route.calls[0].request.content)
    assert body["input"][0] == {"role": "developer", "content": "Be brief."}
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-openai"


@respx.mock
async def test_reasoning_signature_round_trip(models):
    route = respx.post(OPENAI_ENDPOINT)
    route.side_effect = [
        httpx.Response(200, text=reasoning_turn()),
        httpx.Response(200, text=text_turn("Yes.")),
    ]

    ctx = Context(messages=[user_text("what is 2+2?")])
    first = await models.complete_simple(gpt(models), ctx, reasoning="high")

    assert first.content[0].type == "thinking"
    assert first.content[0].thinking == "thinking hard"

    ctx.messages.append(first)
    ctx.messages.append(user_text("are you sure?"))
    await models.complete_simple(gpt(models), ctx, reasoning="high")

    # 推理 item 必须整个回放，加密内容要在里面
    body = json.loads(route.calls[1].request.content)
    reasoning_item = next(i for i in body["input"] if i.get("type") == "reasoning")
    assert reasoning_item["encrypted_content"] == "blob"


@respx.mock
async def test_full_tool_calling_round_trip(models):
    route = respx.post(OPENAI_ENDPOINT)
    route.side_effect = [
        httpx.Response(200, text=tool_turn("get_time", '{"tz":"UTC"}')),
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

    first = await models.complete(gpt(models), ctx)
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

    second = await models.complete(gpt(models), ctx)
    assert second.content[0].text == "It is noon."

    body = json.loads(route.calls[1].request.content)
    kinds = [i.get("type") or i.get("role") for i in body["input"]]
    assert kinds == ["user", "function_call", "function_call_output"]
    assert body["input"][1]["call_id"] == "call_1"
    assert body["input"][2]["output"] == "12:00 UTC"


@respx.mock
async def test_azure_end_to_end(models):
    route = respx.post(AZURE_ENDPOINT).mock(return_value=httpx.Response(200, text=text_turn("Azure ok")))

    azure_model = models.get_models("azure-openai-responses")[0]
    message = await models.complete(azure_model, Context(messages=[user_text("hi")]))

    assert message.content[0].text == "Azure ok"
    request = route.calls[0].request
    assert request.headers["api-key"] == "sk-azure"
    assert request.url.params["api-version"] == "v1"


@respx.mock
async def test_azure_reports_missing_endpoint_config_as_a_message(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-azure")
    collection = create_models()
    collection.set_provider(azure_openai_responses_provider())

    model = collection.get_models("azure-openai-responses")[0]
    message = await collection.complete(model, Context(messages=[user_text("hi")]))

    assert message.stop_reason == "error"
    assert "AZURE_OPENAI_BASE_URL" in message.error_message


@respx.mock
async def test_errors_surface_as_messages(models):
    respx.post(OPENAI_ENDPOINT).mock(
        return_value=httpx.Response(500, text='{"error":{"message":"upstream exploded"}}')
    )

    message = await models.complete(gpt(models), Context(messages=[user_text("hi")]))
    assert message.stop_reason == "error"
    assert "upstream exploded" in message.error_message


@respx.mock
async def test_all_four_protocols_share_one_collection(monkeypatch):
    """25 家 + 2 家 Responses 厂商可以注册进同一个集合。"""
    from pi_ai.providers import anthropic_messages_providers, openai_completions_providers

    collection = create_models()
    for provider in [
        *openai_completions_providers(),
        *anthropic_messages_providers(),
        *responses_providers(),
    ]:
        collection.set_provider(provider)

    assert len(collection.get_providers()) == 27
    apis = {m.api for m in collection.get_models()}
    assert apis == {"openai-completions", "anthropic-messages", "openai-responses", "azure-openai-responses"}
