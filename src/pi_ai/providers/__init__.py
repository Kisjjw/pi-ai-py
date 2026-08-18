from __future__ import annotations

from pi_ai.api.anthropic_messages import anthropic_messages_api
from pi_ai.api.azure_openai_responses import azure_openai_responses_api
from pi_ai.api.google_generative_ai import google_generative_ai_api
from pi_ai.api.openai_completions import openai_completions_api
from pi_ai.api.openai_responses import openai_responses_api
from pi_ai.auth.env import env_api_key_auth
from pi_ai.catalog import load_provider_models
from pi_ai.provider import Provider, ProviderStreams, create_provider
from pi_ai.providers._anthropic_table import (
    ANTHROPIC_COMPATIBLE_SPECS,
    ANTHROPIC_SPEC,
    AnthropicApiKeyAuth,
    AnthropicProviderSpec,
)
from pi_ai.providers._cloudflare import (
    CloudflareAIGatewayAuth,
    CloudflareWorkersAIAuth,
    cloudflare_streams,
)
from pi_ai.providers._mixed_table import (
    CLOUDFLARE_SPECS,
    CLOUDFLARE_SPECS_BY_ID,
    MIXED_PROVIDER_SPECS,
    MIXED_PROVIDER_SPECS_BY_ID,
    MixedProviderSpec,
)
from pi_ai.providers._table import PROVIDER_SPECS, PROVIDER_SPECS_BY_ID, ProviderSpec
from pi_ai.providers.faux import (
    FauxApiKeyAuth,
    FauxRequest,
    FauxResponse,
    FauxStreams,
    FauxToolCall,
    faux_model,
    faux_provider,
)
from pi_ai.types import Model

_ANTHROPIC_SPECS_BY_ID = {spec.id: spec for spec in ANTHROPIC_COMPATIBLE_SPECS}


def _api_by_id(api_id: str) -> ProviderStreams:
    if api_id == "anthropic-messages":
        return anthropic_messages_api()
    if api_id == "openai-completions":
        return openai_completions_api()
    if api_id == "openai-responses":
        return openai_responses_api()
    if api_id == "google-generative-ai":
        return google_generative_ai_api()
    raise LookupError(f"unknown api {api_id!r}")


def _spec_base_url(spec: MixedProviderSpec, models: list[Model]) -> str:
    """opencode 这类厂商没有 provider 级 base_url，取目录里第一个模型的。"""
    if spec.base_url is not None:
        return spec.base_url
    return models[0].base_url if models else ""


def _build(spec: ProviderSpec) -> Provider:
    return create_provider(
        id=spec.id,
        name=spec.name,
        base_url=spec.base_url,
        auth={"api_key": env_api_key_auth(spec.auth_name, spec.env_vars)},
        models=list(load_provider_models(spec.id).values()),
        api=openai_completions_api(),
    )


def _build_anthropic_compatible(spec: AnthropicProviderSpec) -> Provider:
    return create_provider(
        id=spec.id,
        name=spec.name,
        base_url=spec.base_url,
        auth={"api_key": env_api_key_auth(spec.auth_name, spec.env_vars)},
        models=list(load_provider_models(spec.id).values()),
        api=anthropic_messages_api(),
    )


def _build_mixed(spec: MixedProviderSpec) -> Provider:
    models = list(load_provider_models(spec.id).values())
    # GitHub Copilot 在 TS 侧还有个 filterModels：按 OAuth 凭据里的 availableModelIds
    # 过滤模型。OAuth 登录不在本层范围内，用环境变量 token 时该字段不存在，
    # TS 的实现也等价于不过滤，故此处不实现。
    return create_provider(
        id=spec.id,
        name=spec.name,
        base_url=_spec_base_url(spec, models),
        auth={"api_key": env_api_key_auth(spec.auth_name, spec.env_vars)},
        models=models,
        api={api_id: _api_by_id(api_id) for api_id in spec.apis},
    )


def _build_cloudflare(spec: MixedProviderSpec) -> Provider:
    models = list(load_provider_models(spec.id).values())
    auth = CloudflareAIGatewayAuth() if spec.id == "cloudflare-ai-gateway" else CloudflareWorkersAIAuth()
    return create_provider(
        id=spec.id,
        name=spec.name,
        base_url=_spec_base_url(spec, models),
        auth={"api_key": auth},
        models=models,
        api={api_id: cloudflare_streams(_api_by_id(api_id)) for api_id in spec.apis},
    )


def provider_by_id(provider_id: str) -> Provider:
    """按 id 构造一家厂商。未知 id 抛 KeyError。"""
    if provider_id == ANTHROPIC_SPEC.id:
        return anthropic_provider()
    if provider_id == "openai":
        return openai_provider()
    if provider_id == "azure-openai-responses":
        return azure_openai_responses_provider()
    if provider_id == "google":
        return google_provider()
    if provider_id in _ANTHROPIC_SPECS_BY_ID:
        return _build_anthropic_compatible(_ANTHROPIC_SPECS_BY_ID[provider_id])
    if provider_id in MIXED_PROVIDER_SPECS_BY_ID:
        return _build_mixed(MIXED_PROVIDER_SPECS_BY_ID[provider_id])
    if provider_id in CLOUDFLARE_SPECS_BY_ID:
        return _build_cloudflare(CLOUDFLARE_SPECS_BY_ID[provider_id])
    return _build(PROVIDER_SPECS_BY_ID[provider_id])


def openai_provider() -> Provider:
    return create_provider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        auth={"api_key": env_api_key_auth("OpenAI API key", ["OPENAI_API_KEY"])},
        models=list(load_provider_models("openai").values()),
        api=openai_responses_api(),
    )


def azure_openai_responses_provider() -> Provider:
    """Azure 的端点在运行时由环境变量拼出，provider 级 base_url 留空。"""
    return create_provider(
        id="azure-openai-responses",
        name="Azure OpenAI",
        base_url="",
        auth={"api_key": env_api_key_auth("Azure OpenAI API key", ["AZURE_OPENAI_API_KEY"])},
        models=list(load_provider_models("azure-openai-responses").values()),
        api=azure_openai_responses_api(),
    )


def responses_providers() -> list[Provider]:
    """构造全部走 Responses 协议的厂商。"""
    return [openai_provider(), azure_openai_responses_provider()]


def google_provider() -> Provider:
    return create_provider(
        id="google",
        name="Google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        auth={"api_key": env_api_key_auth("Google Gemini API key", ["GEMINI_API_KEY"])},
        models=list(load_provider_models("google").values()),
        api=google_generative_ai_api(),
    )


def openai_completions_providers() -> list[Provider]:
    """构造全部走 openai-completions 协议且认证为标准环境变量的厂商。"""
    return [_build(spec) for spec in PROVIDER_SPECS]


def anthropic_messages_providers() -> list[Provider]:
    """构造全部走 anthropic-messages 协议的厂商。"""
    return [anthropic_provider(), *(_build_anthropic_compatible(s) for s in ANTHROPIC_COMPATIBLE_SPECS)]


def anthropic_provider() -> Provider:
    return create_provider(
        id=ANTHROPIC_SPEC.id,
        name=ANTHROPIC_SPEC.name,
        base_url=ANTHROPIC_SPEC.base_url,
        auth={"api_key": AnthropicApiKeyAuth()},
        models=list(load_provider_models(ANTHROPIC_SPEC.id).values()),
        api=anthropic_messages_api(),
    )


def mixed_providers() -> list[Provider]:
    """构造全部混合协议厂商（含两个 Cloudflare）。"""
    return [
        *(_build_mixed(spec) for spec in MIXED_PROVIDER_SPECS),
        *(_build_cloudflare(spec) for spec in CLOUDFLARE_SPECS),
    ]


def all_providers() -> list[Provider]:
    """构造全部 35 家厂商。"""
    return [
        *openai_completions_providers(),
        *anthropic_messages_providers(),
        *responses_providers(),
        google_provider(),
        *mixed_providers(),
    ]


def xai_provider() -> Provider:
    return provider_by_id("xai")


def fireworks_provider() -> Provider:
    return provider_by_id("fireworks")


def github_copilot_provider() -> Provider:
    return provider_by_id("github-copilot")


def opencode_provider() -> Provider:
    return provider_by_id("opencode")


def opencode_go_provider() -> Provider:
    return provider_by_id("opencode-go")


def cloudflare_ai_gateway_provider() -> Provider:
    return provider_by_id("cloudflare-ai-gateway")


def cloudflare_workers_ai_provider() -> Provider:
    return provider_by_id("cloudflare-workers-ai")


def kimi_coding_provider() -> Provider:
    return provider_by_id("kimi-coding")


def minimax_provider() -> Provider:
    return provider_by_id("minimax")


def minimax_cn_provider() -> Provider:
    return provider_by_id("minimax-cn")


def vercel_ai_gateway_provider() -> Provider:
    return provider_by_id("vercel-ai-gateway")


def ant_ling_provider() -> Provider:
    return provider_by_id("ant-ling")


def baseten_provider() -> Provider:
    return provider_by_id("baseten")


def cerebras_provider() -> Provider:
    return provider_by_id("cerebras")


def deepseek_provider() -> Provider:
    return provider_by_id("deepseek")


def groq_provider() -> Provider:
    return provider_by_id("groq")


def huggingface_provider() -> Provider:
    return provider_by_id("huggingface")


def moonshotai_provider() -> Provider:
    return provider_by_id("moonshotai")


def moonshotai_cn_provider() -> Provider:
    return provider_by_id("moonshotai-cn")


def nvidia_provider() -> Provider:
    return provider_by_id("nvidia")


def openrouter_provider() -> Provider:
    return provider_by_id("openrouter")


def qwen_token_plan_provider() -> Provider:
    return provider_by_id("qwen-token-plan")


def qwen_token_plan_cn_provider() -> Provider:
    return provider_by_id("qwen-token-plan-cn")


def qwen_token_plan_individual_provider() -> Provider:
    return provider_by_id("qwen-token-plan-individual")


def together_provider() -> Provider:
    return provider_by_id("together")


def xiaomi_provider() -> Provider:
    return provider_by_id("xiaomi")


def xiaomi_token_plan_cn_provider() -> Provider:
    return provider_by_id("xiaomi-token-plan-cn")


def xiaomi_token_plan_ams_provider() -> Provider:
    return provider_by_id("xiaomi-token-plan-ams")


def xiaomi_token_plan_sgp_provider() -> Provider:
    return provider_by_id("xiaomi-token-plan-sgp")


def zai_provider() -> Provider:
    return provider_by_id("zai")


def zai_coding_cn_provider() -> Provider:
    return provider_by_id("zai-coding-cn")


__all__ = [
    "ANTHROPIC_COMPATIBLE_SPECS",
    "ANTHROPIC_SPEC",
    "CLOUDFLARE_SPECS",
    "MIXED_PROVIDER_SPECS",
    "PROVIDER_SPECS",
    "AnthropicProviderSpec",
    "FauxApiKeyAuth",
    "FauxRequest",
    "FauxResponse",
    "FauxStreams",
    "FauxToolCall",
    "MixedProviderSpec",
    "ProviderSpec",
    "all_providers",
    "ant_ling_provider",
    "anthropic_messages_providers",
    "anthropic_provider",
    "azure_openai_responses_provider",
    "baseten_provider",
    "cerebras_provider",
    "cloudflare_ai_gateway_provider",
    "cloudflare_workers_ai_provider",
    "deepseek_provider",
    "faux_model",
    "faux_provider",
    "fireworks_provider",
    "github_copilot_provider",
    "google_provider",
    "groq_provider",
    "huggingface_provider",
    "kimi_coding_provider",
    "minimax_cn_provider",
    "minimax_provider",
    "mixed_providers",
    "moonshotai_cn_provider",
    "moonshotai_provider",
    "nvidia_provider",
    "openai_completions_providers",
    "openai_provider",
    "opencode_go_provider",
    "opencode_provider",
    "openrouter_provider",
    "provider_by_id",
    "responses_providers",
    "qwen_token_plan_cn_provider",
    "qwen_token_plan_individual_provider",
    "qwen_token_plan_provider",
    "together_provider",
    "vercel_ai_gateway_provider",
    "xai_provider",
    "xiaomi_provider",
    "xiaomi_token_plan_ams_provider",
    "xiaomi_token_plan_cn_provider",
    "xiaomi_token_plan_sgp_provider",
    "zai_coding_cn_provider",
    "zai_provider",
]
