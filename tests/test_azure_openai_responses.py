import json

import httpx
import pytest
import respx

from pi_ai.api.azure_openai_responses import (
    build_params,
    build_request_headers,
    build_request_url,
    normalize_azure_base_url,
    parse_deployment_name_map,
    resolve_azure_config,
    resolve_deployment_name,
    stream,
    stream_simple,
)
from pi_ai.http.errors import ProviderHTTPError
from pi_ai.types import Context, Model, ModelCost, Tool, user_text

RESOURCE_ENDPOINT = "https://my-resource.openai.azure.com/openai/v1/responses"


def make_model(*, model_id="gpt-5", base_url="", reasoning=True, compat=None, level_map=None) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="azure-openai-responses",
        provider="azure-openai-responses",
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
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "m"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": text},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "message", "id": "m", "content": [{"type": "output_text", "text": text}]},
        },
        {
            "type": "response.completed",
            "response": {
                "id": "r",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
    )


# --- base url 归一化 ---


@pytest.mark.parametrize(
    "given",
    [
        "https://my-resource.openai.azure.com",
        "https://my-resource.openai.azure.com/",
        "https://my-resource.openai.azure.com/openai",
        "https://my-resource.openai.azure.com/openai/v1",
        "https://my-resource.openai.azure.com/openai/v1/",
        "https://my-resource.openai.azure.com/openai/v1/responses",
    ],
)
def test_azure_root_endpoints_normalize_to_openai_v1(given):
    assert normalize_azure_base_url(given) == "https://my-resource.openai.azure.com/openai/v1"


@pytest.mark.parametrize(
    "host",
    ["my.openai.azure.com", "my.cognitiveservices.azure.com", "my.ai.azure.com"],
)
def test_all_three_azure_host_suffixes_are_recognized(host):
    assert normalize_azure_base_url(f"https://{host}") == f"https://{host}/openai/v1"


def test_query_is_stripped_when_rewriting():
    assert (
        normalize_azure_base_url("https://my.openai.azure.com/openai?api-version=2024-01")
        == "https://my.openai.azure.com/openai/v1"
    )


def test_custom_paths_on_azure_hosts_are_left_alone():
    given = "https://my.openai.azure.com/custom/path"
    assert normalize_azure_base_url(given) == given


def test_non_azure_hosts_are_left_alone():
    given = "https://proxy.example.com/v1"
    assert normalize_azure_base_url(given) == given
    assert normalize_azure_base_url("https://proxy.example.com") == "https://proxy.example.com"


def test_invalid_urls_raise():
    with pytest.raises(ProviderHTTPError, match="Invalid Azure OpenAI base URL"):
        normalize_azure_base_url("not-a-url")


# --- 配置解析 ---


def test_explicit_base_url_wins():
    base_url, version = resolve_azure_config(
        make_model(), {"azure_base_url": "https://explicit.openai.azure.com"}
    )
    assert base_url == "https://explicit.openai.azure.com/openai/v1"
    assert version == "v1"


def test_env_base_url_is_used(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://from-env.openai.azure.com")
    assert resolve_azure_config(make_model())[0] == "https://from-env.openai.azure.com/openai/v1"


def test_resource_name_builds_the_default_endpoint():
    base_url, _ = resolve_azure_config(make_model(), {"azure_resource_name": "my-resource"})
    assert base_url == "https://my-resource.openai.azure.com/openai/v1"


def test_model_base_url_is_the_last_fallback():
    model = make_model(base_url="https://from-model.openai.azure.com")
    assert resolve_azure_config(model)[0] == "https://from-model.openai.azure.com/openai/v1"


def test_missing_configuration_raises_a_helpful_error():
    with pytest.raises(ProviderHTTPError, match="AZURE_OPENAI_BASE_URL"):
        resolve_azure_config(make_model())


def test_api_version_can_be_overridden(monkeypatch):
    explicit = {"azure_base_url": "https://x.openai.azure.com", "azure_api_version": "2025-01"}
    assert resolve_azure_config(make_model(), explicit)[1] == "2025-01"

    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "from-env")
    assert resolve_azure_config(make_model(), {"azure_base_url": "https://x.openai.azure.com"})[1] == "from-env"


def test_request_url_carries_the_api_version():
    assert (
        build_request_url("https://x.openai.azure.com/openai/v1", "v1")
        == "https://x.openai.azure.com/openai/v1/responses?api-version=v1"
    )


# --- 部署名 ---


def test_deployment_map_parsing():
    assert parse_deployment_name_map("a=1,b=2") == {"a": "1", "b": "2"}
    assert parse_deployment_name_map(" a = 1 , b = 2 ") == {"a": "1", "b": "2"}
    assert parse_deployment_name_map("") == {}
    assert parse_deployment_name_map(None) == {}
    assert parse_deployment_name_map("bad,a=1") == {"a": "1"}


def test_deployment_name_resolution_order(monkeypatch):
    model = make_model(model_id="gpt-5")
    assert resolve_deployment_name(model) == "gpt-5"

    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "gpt-5=prod-deployment")
    assert resolve_deployment_name(model) == "prod-deployment"

    assert resolve_deployment_name(model, {"azure_deployment_name": "explicit"}) == "explicit"


def test_request_body_uses_the_deployment_name():
    params = build_params(make_model(model_id="gpt-5"), CTX, {}, "my-deployment")
    assert params["model"] == "my-deployment"


# --- 请求体与请求头 ---


def test_azure_uses_the_api_key_header():
    headers = build_request_headers(make_model(), "sk-azure")
    assert headers["api-key"] == "sk-azure"
    assert "authorization" not in headers


def test_caller_headers_win():
    model = make_model()
    model.headers = {"x-a": "model"}
    headers = build_request_headers(model, "sk-1", {"x-a": "caller"})
    assert headers["x-a"] == "caller"


def test_core_body_fields():
    params = build_params(make_model(), CTX, {})
    assert params["stream"] is True
    assert params["store"] is False
    assert params["input"][0]["role"] == "user"


def test_prompt_cache_key_is_sent_unconditionally():
    """Azure 这边没有 retention 分支，有 sessionId 就发。"""
    assert build_params(make_model(), CTX, {"session_id": "s1"})["prompt_cache_key"] == "s1"


def test_max_output_tokens_floor():
    assert build_params(make_model(), CTX, {"max_tokens": 1})["max_output_tokens"] == 16


def test_tools_default_to_strict_mode_support():
    """Azure 的 strict 默认是 True，与 openai-responses 的 False 相反。"""
    ctx = Context(messages=[user_text("hi")], tools=[Tool(name="t", description="d", parameters={})])
    assert build_params(make_model(), ctx, {})["tools"][0]["strict"] is False

    model = make_model(compat={"supportsStrictMode": False})
    assert "strict" not in build_params(model, ctx, {})["tools"][0]


def test_reasoning_fields():
    params = build_params(make_model(), CTX, {"reasoning_effort": "high"})
    assert params["reasoning"] == {"effort": "high", "summary": "auto"}
    assert params["include"] == ["reasoning.encrypted_content"]

    assert build_params(make_model(), CTX, {})["reasoning"] == {"effort": "none"}
    assert "reasoning" not in build_params(make_model(level_map={"off": None}), CTX, {})


# --- 端到端 ---


@respx.mock
async def test_stream_hits_the_resolved_endpoint():
    route = respx.post(RESOURCE_ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    message = await stream(
        make_model(), CTX, {"api_key": "sk-azure", "azure_resource_name": "my-resource"}
    ).result()

    assert message.content[0].text == "Hello"
    request = route.calls[0].request
    assert request.headers["api-key"] == "sk-azure"
    assert request.url.params["api-version"] == "v1"


@respx.mock
async def test_deployment_map_reaches_the_wire(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "gpt-5=prod")
    route = respx.post(RESOURCE_ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream(make_model(), CTX, {"api_key": "sk", "azure_resource_name": "my-resource"}).result()

    assert json.loads(route.calls[0].request.content)["model"] == "prod"


async def test_missing_api_key_becomes_an_error_event():
    message = await stream(make_model(), CTX, {"azure_resource_name": "r"}).result()
    assert message.stop_reason == "error"
    assert "No API key" in message.error_message


async def test_missing_endpoint_config_becomes_an_error_event():
    message = await stream(make_model(), CTX, {"api_key": "sk"}).result()
    assert message.stop_reason == "error"
    assert "AZURE_OPENAI_BASE_URL" in message.error_message


@respx.mock
async def test_http_errors_are_prefixed():
    respx.post(RESOURCE_ENDPOINT).mock(return_value=httpx.Response(401, text='{"error":{"message":"nope"}}'))

    message = await stream(
        make_model(), CTX, {"api_key": "sk", "azure_resource_name": "my-resource"}
    ).result()
    assert "Azure OpenAI API error" in message.error_message


@respx.mock
async def test_stream_simple_translates_reasoning():
    route = respx.post(RESOURCE_ENDPOINT).mock(return_value=httpx.Response(200, text=text_response()))

    await stream_simple(
        make_model(), CTX, {"api_key": "sk", "azure_resource_name": "my-resource", "reasoning": "low"}
    ).result()

    assert json.loads(route.calls[0].request.content)["reasoning"]["effort"] == "low"
