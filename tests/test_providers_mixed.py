"""混合协议厂商：按 model.api 分派，以及 Cloudflare 的端点与鉴权头。"""

import json

import httpx
import pytest
import respx

from pi_ai import Context, create_models, user_text
from pi_ai.providers import (
    CLOUDFLARE_SPECS,
    MIXED_PROVIDER_SPECS,
    all_providers,
    cloudflare_ai_gateway_provider,
    cloudflare_workers_ai_provider,
    fireworks_provider,
    github_copilot_provider,
    mixed_providers,
    opencode_go_provider,
    opencode_provider,
    provider_by_id,
    xai_provider,
)

ENV_VARS = (
    "XAI_API_KEY",
    "FIREWORKS_API_KEY",
    "COPILOT_GITHUB_TOKEN",
    "OPENCODE_API_KEY",
    "CLOUDFLARE_API_KEY",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_GATEWAY_ID",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def completions_sse(text: str) -> str:
    return (
        'data: {"id":"r1","choices":[{"index":0,"delta":{"role":"assistant","content":"'
        + text
        + '"}}]}\n\n'
        'data: {"id":"r1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n'
        "data: [DONE]\n\n"
    )


def sse(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def anthropic_sse(text: str) -> str:
    return sse(
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    )


def responses_sse(text: str) -> str:
    return sse(
        {"type": "response.created", "response": {"id": "resp_1"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_1"},
        },
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
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        },
    )


# --- 注册与目录 ---


@pytest.mark.parametrize("spec", MIXED_PROVIDER_SPECS, ids=lambda s: s.id)
def test_mixed_provider_registers_every_api_it_declares(spec):
    provider = provider_by_id(spec.id)
    assert provider.name == spec.name
    assert isinstance(provider.api, dict)
    assert set(provider.api) == set(spec.apis)

    models = provider.get_models()
    assert models
    assert all(m.provider == spec.id for m in models)
    # 目录里出现的每种协议都要有对应实现
    assert {m.api for m in models} <= set(spec.apis)


@pytest.mark.parametrize("spec", MIXED_PROVIDER_SPECS, ids=lambda s: s.id)
def test_every_model_can_be_dispatched(spec):
    provider = provider_by_id(spec.id)
    for model in provider.get_models():
        assert provider.streams_for(model) is not None


def test_mixed_providers_builds_all_seven():
    ids = [p.id for p in mixed_providers()]
    assert ids == [s.id for s in (*MIXED_PROVIDER_SPECS, *CLOUDFLARE_SPECS)]


def test_opencode_covers_all_four_protocols():
    provider = opencode_provider()
    assert set(provider.api) == {
        "anthropic-messages",
        "google-generative-ai",
        "openai-completions",
        "openai-responses",
    }


def test_provider_base_urls():
    assert xai_provider().base_url == "https://api.x.ai/v1"
    assert fireworks_provider().base_url == "https://api.fireworks.ai/inference"
    assert github_copilot_provider().base_url == "https://api.individual.githubcopilot.com"
    # 目录里没有 provider 级 base_url 的，退回第一个模型的
    assert opencode_provider().base_url == opencode_provider().get_models()[0].base_url
    assert opencode_go_provider().base_url == opencode_go_provider().get_models()[0].base_url


def test_unknown_api_raises():
    provider = fireworks_provider()
    model = provider.get_models()[0].model_copy(update={"api": "openai-responses"})
    with pytest.raises(LookupError):
        provider.streams_for(model)


# --- 认证 ---


@pytest.mark.parametrize("spec", MIXED_PROVIDER_SPECS, ids=lambda s: s.id)
async def test_mixed_provider_auth_resolves_from_env(spec, monkeypatch):
    monkeypatch.setenv(spec.env_vars[0], "sk-test")
    models = create_models()
    models.set_provider(provider_by_id(spec.id))
    resolved = await models.get_auth(spec.id)
    assert resolved is not None
    assert resolved.auth.api_key == "sk-test"


async def test_opencode_and_opencode_go_share_one_env_var(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-oc")
    models = create_models()
    models.set_provider(opencode_provider())
    models.set_provider(opencode_go_provider())
    for provider_id in ("opencode", "opencode-go"):
        resolved = await models.get_auth(provider_id)
        assert resolved is not None
        assert resolved.auth.api_key == "sk-oc"


# --- 分派到正确的适配器 ---


@respx.mock
async def test_fireworks_dispatches_by_model_api(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "sk-fw")
    models = create_models()
    models.set_provider(fireworks_provider())

    completions = next(m for m in models.get_models("fireworks") if m.api == "openai-completions")
    anthropic = next(
        (m for m in models.get_models("fireworks") if m.api == "anthropic-messages"), None
    )

    completions_route = respx.post(f"{completions.base_url}/chat/completions").mock(
        return_value=httpx.Response(200, text=completions_sse("from completions"))
    )
    reply = await models.complete(completions, Context(messages=[user_text("hi")]))
    assert reply.content[0].text == "from completions"
    assert completions_route.calls[0].request.headers["authorization"] == "Bearer sk-fw"

    if anthropic is not None:
        anthropic_route = respx.post(f"{anthropic.base_url}/v1/messages").mock(
            return_value=httpx.Response(200, text=anthropic_sse("from anthropic"))
        )
        reply = await models.complete(anthropic, Context(messages=[user_text("hi")]))
        assert reply.content[0].text == "from anthropic"
        assert anthropic_route.calls[0].request.headers["x-api-key"] == "sk-fw"


@respx.mock
async def test_xai_dispatches_completions_and_responses(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "sk-xai")
    models = create_models()
    models.set_provider(xai_provider())

    for model in models.get_models("xai"):
        if model.api == "openai-completions":
            respx.post(f"{model.base_url}/chat/completions").mock(
                return_value=httpx.Response(200, text=completions_sse("c"))
            )
        else:
            respx.post(f"{model.base_url}/responses").mock(
                return_value=httpx.Response(200, text=responses_sse("r"))
            )
        reply = await models.complete(model, Context(messages=[user_text("hi")]))
        assert reply.stop_reason == "stop"


# --- Cloudflare 端到端 ---


@respx.mock
async def test_workers_ai_materializes_the_account_placeholder(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")

    models = create_models()
    models.set_provider(cloudflare_workers_ai_provider())
    model = models.get_models("cloudflare-workers-ai")[0]
    assert "{CLOUDFLARE_ACCOUNT_ID}" in model.base_url

    url = "https://api.cloudflare.com/client/v4/accounts/acct-1/ai/v1/chat/completions"
    route = respx.post(url).mock(return_value=httpx.Response(200, text=completions_sse("hello")))

    reply = await models.complete(model, Context(messages=[user_text("hi")]))
    assert reply.content[0].text == "hello"
    assert route.calls[0].request.headers["authorization"] == "Bearer cf-key"


@respx.mock
async def test_ai_gateway_sends_the_cf_header_and_drops_the_default_one(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gw-1")

    models = create_models()
    models.set_provider(cloudflare_ai_gateway_provider())
    model = next(m for m in models.get_models("cloudflare-ai-gateway") if m.api == "anthropic-messages")

    url = "https://gateway.ai.cloudflare.com/v1/acct-1/gw-1/anthropic/v1/messages"
    route = respx.post(url).mock(return_value=httpx.Response(200, text=anthropic_sse("via gateway")))

    reply = await models.complete(model, Context(messages=[user_text("hi")]))
    assert reply.content[0].text == "via gateway"

    headers = route.calls[0].request.headers
    assert headers["cf-aig-authorization"] == "Bearer cf-key"
    # 网关只认 cf-aig-authorization，默认鉴权头必须被删掉
    assert "authorization" not in headers
    assert "x-api-key" not in headers


@respx.mock
async def test_ai_gateway_dispatches_every_protocol(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gw-1")

    models = create_models()
    models.set_provider(cloudflare_ai_gateway_provider())
    base = "https://gateway.ai.cloudflare.com/v1/acct-1/gw-1"

    respx.post(f"{base}/anthropic/v1/messages").mock(
        return_value=httpx.Response(200, text=anthropic_sse("a"))
    )
    respx.post(f"{base}/compat/chat/completions").mock(
        return_value=httpx.Response(200, text=completions_sse("c"))
    )
    respx.post(f"{base}/openai/responses").mock(
        return_value=httpx.Response(200, text=responses_sse("r"))
    )

    seen = set()
    for model in models.get_models("cloudflare-ai-gateway"):
        if model.api in seen:
            continue
        seen.add(model.api)
        reply = await models.complete(model, Context(messages=[user_text("hi")]))
        assert reply.stop_reason == "stop", model.id

    assert seen == {"anthropic-messages", "openai-completions", "openai-responses"}


async def test_cloudflare_is_unconfigured_without_the_account_id(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    models = create_models()
    models.set_provider(cloudflare_workers_ai_provider())
    assert await models.get_auth("cloudflare-workers-ai") is None


@respx.mock
async def test_unconfigured_cloudflare_surfaces_as_a_message():
    models = create_models()
    models.set_provider(cloudflare_workers_ai_provider())
    model = models.get_models("cloudflare-workers-ai")[0]

    message = await models.complete(model, Context(messages=[user_text("hi")]))
    assert message.stop_reason == "error"
    assert "not configured" in message.error_message


# --- 全量集合 ---


def test_all_thirty_five_providers_share_one_collection():
    collection = create_models()
    for provider in all_providers():
        collection.set_provider(provider)

    assert len(collection.get_providers()) == 35
    assert {m.api for m in collection.get_models()} == {
        "openai-completions",
        "anthropic-messages",
        "openai-responses",
        "azure-openai-responses",
        "google-generative-ai",
    }


def test_every_model_in_the_full_collection_can_be_dispatched():
    for provider in all_providers():
        for model in provider.get_models():
            assert provider.streams_for(model) is not None, f"{provider.id}/{model.id}"
