from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse, urlunparse

from pi_ai.api.constrained_sampling import create_grammar_tool_input_properties
from pi_ai.api.openai_prompt_cache import clamp_openai_prompt_cache_key
from pi_ai.api.openai_responses import (
    OPENAI_RESPONSES_MIN_OUTPUT_TOKENS,
    consume_responses_stream,
    emit_responses_error,
)
from pi_ai.api.openai_responses_shared import convert_responses_messages, convert_responses_tools
from pi_ai.api.openai_responses_stream import ResponsesStreamParser
from pi_ai.api.simple_options import build_base_options
from pi_ai.cost import clamp_thinking_level
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.http.client import merge_headers
from pi_ai.http.errors import ProviderHTTPError
from pi_ai.types import Context, Model
from pi_ai.utils.provider_env import get_provider_env_value

DEFAULT_AZURE_API_VERSION = "v1"

#: Azure 端点也接受 OpenAI Responses 风格的复合工具调用 id。
AZURE_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode", "azure-openai-responses"})

_AZURE_HOST_SUFFIXES = (".openai.azure.com", ".cognitiveservices.azure.com", ".ai.azure.com")

#: 这些路径说明用户给的是根端点，需要补成 /openai/v1。
_REWRITABLE_PATHS = ("", "/", "/openai", "/openai/v1/responses")

_MISSING = object()


def parse_deployment_name_map(value: str | None) -> dict[str, str]:
    """解析 `model-id=deployment,model-id2=deployment2` 形式的映射。"""
    mapping: dict[str, str] = {}
    if not value:
        return mapping
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        model_id, separator, deployment = trimmed.partition("=")
        if not separator or not model_id.strip() or not deployment.strip():
            continue
        mapping[model_id.strip()] = deployment.strip()
    return mapping


def resolve_deployment_name(model: Model, options: Mapping[str, Any] | None = None) -> str:
    """请求体里的 model 字段用的是部署名，不是模型 id。"""
    options = options or {}
    explicit = options.get("azure_deployment_name")
    if explicit:
        return str(explicit)
    mapped = parse_deployment_name_map(
        get_provider_env_value("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", options.get("env"))
    ).get(model.id)
    return mapped or model.id


def normalize_azure_base_url(base_url: str) -> str:
    """把各种形态的 Azure 端点归一成带 /openai/v1 的基址。

    用户常见的输入是资源根端点（https://x.openai.azure.com），也可能直接
    粘了完整的 responses 路径。两者都要收敛到同一个基址，后面才能稳定地
    拼出 /responses。
    """
    trimmed = base_url.strip().rstrip("/")
    parsed = urlparse(trimmed)
    if not parsed.scheme or not parsed.netloc:
        raise ProviderHTTPError(f"Invalid Azure OpenAI base URL: {base_url}")

    hostname = (parsed.hostname or "").lower()
    is_azure_host = any(hostname.endswith(suffix) for suffix in _AZURE_HOST_SUFFIXES)
    normalized_path = parsed.path.rstrip("/")

    if is_azure_host and normalized_path in _REWRITABLE_PATHS:
        parsed = parsed._replace(path="/openai/v1", query="", fragment="")

    return urlunparse(parsed).rstrip("/")


def resolve_azure_config(
    model: Model,
    options: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """解析出 (base_url, api_version)。

    Azure 的模型目录里 base_url 是空字符串，端点必须在运行时拼出来。
    """
    options = options or {}
    api_version = (
        options.get("azure_api_version")
        or get_provider_env_value("AZURE_OPENAI_API_VERSION", options.get("env"))
        or DEFAULT_AZURE_API_VERSION
    )

    explicit = options.get("azure_base_url")
    base_url = (explicit or "").strip() or (
        get_provider_env_value("AZURE_OPENAI_BASE_URL", options.get("env")) or ""
    ).strip() or None

    resource_name = options.get("azure_resource_name") or get_provider_env_value(
        "AZURE_OPENAI_RESOURCE_NAME", options.get("env")
    )

    if not base_url and resource_name:
        base_url = f"https://{resource_name}.openai.azure.com/openai/v1"
    if not base_url and model.base_url:
        base_url = model.base_url
    if not base_url:
        raise ProviderHTTPError(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or "
            "AZURE_OPENAI_RESOURCE_NAME, or pass azure_base_url, azure_resource_name, or model.base_url."
        )

    return normalize_azure_base_url(base_url), api_version


def build_request_url(base_url: str, api_version: str) -> str:
    return f"{base_url.rstrip('/')}/responses?api-version={api_version}"


def build_request_headers(
    model: Model,
    api_key: str,
    options_headers: Mapping[str, str | None] | None = None,
) -> dict[str, str]:
    """Azure 用 api-key 头，不是 Authorization。"""
    merged = merge_headers(model.headers, options_headers)
    merged.setdefault("api-key", api_key)
    merged.setdefault("content-type", "application/json")
    merged.setdefault("accept", "text/event-stream")
    return merged


def build_params(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
    deployment_name: str | None = None,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    options = options or {}
    compat = model.compat
    supports_grammar = bool(getattr(compat, "supports_openai_grammar_tools", None))
    supports_strict = getattr(compat, "supports_strict_mode", None)
    if supports_strict is None:
        supports_strict = True

    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(context.tools, supports_grammar)

    messages = convert_responses_messages(
        model,
        context,
        AZURE_TOOL_CALL_PROVIDERS,
        grammar_tool_input_properties=grammar_tool_input_properties,
    )

    params: dict[str, Any] = {
        "model": deployment_name or resolve_deployment_name(model, options),
        "input": messages,
        "stream": True,
        "store": False,
    }

    cache_key = clamp_openai_prompt_cache_key(options.get("session_id"))
    if cache_key is not None:
        params["prompt_cache_key"] = cache_key

    max_tokens = options.get("max_tokens")
    if max_tokens:
        params["max_output_tokens"] = max(max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options.get("temperature") is not None:
        params["temperature"] = options["temperature"]

    if context.tools:
        params["tools"] = convert_responses_tools(
            context.tools,
            supports_strict_mode=supports_strict,
            supports_openai_grammar_tools=supports_grammar,
        )

    _apply_reasoning(params, model, options)

    if options.get("sampling_params"):
        params.update(options["sampling_params"])

    return params


def _apply_reasoning(params: dict[str, Any], model: Model, options: Mapping[str, Any]) -> None:
    if not model.reasoning:
        return

    effort_option = options.get("reasoning_effort")
    summary_option = options.get("reasoning_summary")
    level_map = model.thinking_level_map or {}

    if effort_option or summary_option:
        effort = level_map.get(effort_option, effort_option) if effort_option else "medium"
        params["reasoning"] = {"effort": effort, "summary": summary_option or "auto"}
        params["include"] = ["reasoning.encrypted_content"]
    elif level_map.get("off", _MISSING) is not None:
        off = level_map.get("off", _MISSING)
        params["reasoning"] = {"effort": "none" if off is _MISSING else off}


def stream(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    options = dict(options or {})
    event_stream = AssistantMessageEventStream()

    async def run() -> None:
        compat = model.compat
        supports_grammar = bool(getattr(compat, "supports_openai_grammar_tools", None))
        grammar_properties = create_grammar_tool_input_properties(context.tools, supports_grammar)
        parser = ResponsesStreamParser(model, grammar_tool_input_properties=dict(grammar_properties))
        try:
            api_key = options.get("api_key")
            if not api_key:
                raise ProviderHTTPError(f"No API key for provider: {model.provider}")

            deployment_name = resolve_deployment_name(model, options)
            base_url, api_version = resolve_azure_config(model, options)

            params = build_params(model, context, options, deployment_name, grammar_properties)
            on_payload = options.get("on_payload")
            if on_payload is not None:
                replaced = on_payload(params, model)
                if asyncio.iscoroutine(replaced):
                    replaced = await replaced
                if replaced is not None:
                    params = replaced

            headers = build_request_headers(model, api_key, options.get("headers"))
            url = build_request_url(base_url, api_version)

            await consume_responses_stream(event_stream, parser, model, url, headers, params, options)
        except asyncio.CancelledError:
            emit_responses_error(
                event_stream, parser, aborted=True, error=None, prefix="Azure OpenAI API error"
            )
            raise
        except BaseException as error:  # noqa: BLE001 - 契约要求不抛出
            emit_responses_error(
                event_stream, parser, aborted=False, error=error, prefix="Azure OpenAI API error"
            )

    task = asyncio.ensure_future(run())
    event_stream.bind_producer(task)
    return event_stream


#: Azure 特有的端点配置项。build_base_options 只透传通用字段，这些必须显式带过去。
#: TS 侧 streamSimple 的参数类型是 SimpleStreamOptions，压根传不进这些字段所以不显形；
#: Python 全是 dict，调用方能传也会期待它生效。
_AZURE_OPTION_KEYS = (
    "azure_api_version",
    "azure_base_url",
    "azure_resource_name",
    "azure_deployment_name",
)


def stream_simple(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    options = dict(options or {})
    base = build_base_options(model, context, options, options.get("api_key"))

    reasoning = options.get("reasoning")
    clamped = clamp_thinking_level(model, reasoning) if reasoning else None
    reasoning_effort = None if clamped in (None, "off") else clamped

    azure_options = {key: options[key] for key in _AZURE_OPTION_KEYS if options.get(key) is not None}

    return stream(model, context, {**base, **azure_options, "reasoning_effort": reasoning_effort})


class AzureOpenAIResponsesStreams:
    def stream(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return stream(model, context, options)

    def stream_simple(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return stream_simple(model, context, options)


def azure_openai_responses_api() -> AzureOpenAIResponsesStreams:
    return AzureOpenAIResponsesStreams()
