from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pi_ai.types import Model, OpenAICompletionsCompat

#: thinking_format 的全部合法取值，顺序无意义，仅用于校验。
THINKING_FORMATS = (
    "openai",
    "openrouter",
    "deepseek",
    "together",
    "baseten",
    "zai",
    "qwen",
    "chat-template",
    "qwen-chat-template",
    "string-thinking",
    "ant-ling",
)


@dataclass(frozen=True)
class ResolvedCompat:
    """解析后的兼容开关。每个字段都有确定取值，适配器直接读，不再做 None 判断。"""

    supports_store: bool
    supports_developer_role: bool
    supports_reasoning_effort: bool
    supports_usage_in_streaming: bool
    supports_finish_reason: bool
    max_tokens_field: str
    requires_tool_result_name: bool
    requires_assistant_after_tool_result: bool
    requires_thinking_as_text: bool
    requires_reasoning_content_on_assistant_messages: bool
    thinking_format: str
    zai_tool_stream: bool
    supports_strict_mode: bool
    supports_openai_grammar_tools: bool
    send_session_affinity_headers: bool
    session_affinity_format: str
    supports_long_cache_retention: bool
    open_router_routing: dict[str, Any] = field(default_factory=dict)
    vercel_gateway_routing: dict[str, Any] = field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    chat_template_args: dict[str, Any] = field(default_factory=dict)
    cache_control_format: str | None = None
    deferred_tools_mode: str | None = None
    supports_thinking_token_budget: bool | None = False


def detect_compat(model: Model) -> ResolvedCompat:
    """按 provider 名与 baseUrl 推断兼容设置。

    这是 model.compat 缺省时的基线；显式给出的字段会覆盖这里的推断值。
    移植自 TS 的 openai-completions.ts:1443-1537，判定顺序敏感，勿重排。
    """
    provider = model.provider
    base_url = model.base_url

    is_zai = (
        provider in ("zai", "zai-coding-cn") or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    )
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cloudflare_ai_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url

    is_non_standard = (
        is_nvidia
        or provider == "cerebras"
        or "cerebras.ai" in base_url
        or provider == "xai"
        or "api.x.ai" in base_url
        or is_together
        or "chutes.ai" in base_url
        or "deepseek.com" in base_url
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers_ai
        or is_cloudflare_ai_gateway
        or is_ant_ling
    )

    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url
    use_max_tokens = (
        "chutes.ai" in base_url
        or is_deepseek
        or is_moonshot
        or is_cloudflare_ai_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )

    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_openrouter_developer_role_model = is_openrouter and (
        model.id.startswith("anthropic/") or model.id.startswith("openai/")
    )
    # 这里判的是 provider 字面量，不是 is_openrouter：baseUrl 命中不算数。
    cache_control_format = "anthropic" if provider == "openrouter" and model.id.startswith("anthropic/") else None

    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_together:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    elif is_openrouter:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"

    return ResolvedCompat(
        supports_store=not is_non_standard,
        supports_developer_role=is_openrouter_developer_role_model or (not is_non_standard and not is_openrouter),
        supports_reasoning_effort=not (
            is_grok or is_zai or is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
        supports_usage_in_streaming=True,
        supports_finish_reason=True,
        max_tokens_field="max_tokens" if use_max_tokens else "max_completion_tokens",
        requires_tool_result_name=False,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        requires_reasoning_content_on_assistant_messages=is_deepseek,
        thinking_format=thinking_format,
        open_router_routing={},
        vercel_gateway_routing={},
        chat_template_kwargs={},
        chat_template_args={},
        zai_tool_stream=False,
        supports_thinking_token_budget=False,
        supports_strict_mode=not (is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia),
        supports_openai_grammar_tools=False,
        cache_control_format=cache_control_format,
        send_session_affinity_headers=False,
        deferred_tools_mode=None,
        session_affinity_format="openrouter" if is_openrouter else "openai",
        supports_long_cache_retention=not (
            is_together or is_cloudflare_workers_ai or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
    )


def get_compat(model: Model) -> ResolvedCompat:
    """推断兼容设置后，用 model.compat 里显式给出的字段覆盖。"""
    detected = detect_compat(model)
    compat = model.compat
    if compat is None:
        return detected

    # Model 构造后直接赋值 compat 不会触发 pydantic 校验器，这里补一次解析，
    # 否则所有开关会静默回退到推断值。
    if isinstance(compat, Mapping):
        compat = OpenAICompletionsCompat.model_validate(compat)

    def pick(name: str, fallback: Any) -> Any:
        value = getattr(compat, name, None)
        return fallback if value is None else value

    return ResolvedCompat(
        supports_store=pick("supports_store", detected.supports_store),
        supports_developer_role=pick("supports_developer_role", detected.supports_developer_role),
        supports_reasoning_effort=pick("supports_reasoning_effort", detected.supports_reasoning_effort),
        supports_usage_in_streaming=pick("supports_usage_in_streaming", detected.supports_usage_in_streaming),
        supports_finish_reason=pick("supports_finish_reason", detected.supports_finish_reason),
        max_tokens_field=pick("max_tokens_field", detected.max_tokens_field),
        requires_tool_result_name=pick("requires_tool_result_name", detected.requires_tool_result_name),
        requires_assistant_after_tool_result=pick(
            "requires_assistant_after_tool_result", detected.requires_assistant_after_tool_result
        ),
        requires_thinking_as_text=pick("requires_thinking_as_text", detected.requires_thinking_as_text),
        requires_reasoning_content_on_assistant_messages=pick(
            "requires_reasoning_content_on_assistant_messages",
            detected.requires_reasoning_content_on_assistant_messages,
        ),
        thinking_format=pick("thinking_format", detected.thinking_format),
        # 唯一不回退到推断值的字段，与 TS 的 `?? {}` 一致。
        open_router_routing=getattr(compat, "open_router_routing", None) or {},
        vercel_gateway_routing=pick("vercel_gateway_routing", detected.vercel_gateway_routing),
        chat_template_kwargs=pick("chat_template_kwargs", detected.chat_template_kwargs),
        chat_template_args=pick("chat_template_args", detected.chat_template_args),
        zai_tool_stream=pick("zai_tool_stream", detected.zai_tool_stream),
        supports_thinking_token_budget=pick(
            "supports_thinking_token_budget", detected.supports_thinking_token_budget
        ),
        supports_strict_mode=pick("supports_strict_mode", detected.supports_strict_mode),
        supports_openai_grammar_tools=pick(
            "supports_openai_grammar_tools", detected.supports_openai_grammar_tools
        ),
        cache_control_format=pick("cache_control_format", detected.cache_control_format),
        send_session_affinity_headers=pick(
            "send_session_affinity_headers", detected.send_session_affinity_headers
        ),
        deferred_tools_mode=pick("deferred_tools_mode", detected.deferred_tools_mode),
        session_affinity_format=pick("session_affinity_format", detected.session_affinity_format),
        supports_long_cache_retention=pick(
            "supports_long_cache_retention", detected.supports_long_cache_retention
        ),
    )
