from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MixedProviderSpec:
    """一家混合协议厂商的静态配置。

    apis 列出它用到的协议 id；provider 会把 api 传成映射，由 Provider.streams_for
    按 model.api 分派。base_url 为 None 时取模型目录里第一个模型的 baseUrl
    （TS 侧这几家的 createProvider 也没给 provider 级 baseUrl）。
    """

    id: str
    name: str
    apis: tuple[str, ...]
    auth_name: str
    env_vars: tuple[str, ...]
    base_url: str | None = None


MIXED_PROVIDER_SPECS: tuple[MixedProviderSpec, ...] = (
    MixedProviderSpec(
        id="xai",
        name="xAI",
        apis=("openai-completions", "openai-responses"),
        auth_name="xAI API key",
        env_vars=("XAI_API_KEY",),
        base_url="https://api.x.ai/v1",
    ),
    MixedProviderSpec(
        id="fireworks",
        name="Fireworks",
        apis=("anthropic-messages", "openai-completions"),
        auth_name="Fireworks API key",
        env_vars=("FIREWORKS_API_KEY",),
        base_url="https://api.fireworks.ai/inference",
    ),
    MixedProviderSpec(
        id="github-copilot",
        name="GitHub Copilot",
        apis=("anthropic-messages", "openai-completions", "openai-responses"),
        auth_name="GitHub Copilot token",
        env_vars=("COPILOT_GITHUB_TOKEN",),
        base_url="https://api.individual.githubcopilot.com",
    ),
    MixedProviderSpec(
        id="openrouter",
        name="OpenRouter",
        apis=("anthropic-messages", "openai-completions"),
        auth_name="OpenRouter API key",
        env_vars=("OPENROUTER_API_KEY",),
        base_url="https://openrouter.ai/api/v1",
    ),
    MixedProviderSpec(
        id="opencode",
        name="OpenCode Zen",
        apis=(
            "anthropic-messages",
            "google-generative-ai",
            "openai-completions",
            "openai-responses",
        ),
        auth_name="OpenCode API key",
        env_vars=("OPENCODE_API_KEY",),
    ),
    MixedProviderSpec(
        id="opencode-go",
        name="OpenCode Go",
        apis=("anthropic-messages", "openai-completions", "openai-responses"),
        auth_name="OpenCode API key",
        env_vars=("OPENCODE_API_KEY",),
    ),
)

MIXED_PROVIDER_SPECS_BY_ID = {spec.id: spec for spec in MIXED_PROVIDER_SPECS}

#: 两个 Cloudflare 厂商单列：认证是自定义的，且协议实现要包一层占位符替换。
CLOUDFLARE_SPECS: tuple[MixedProviderSpec, ...] = (
    MixedProviderSpec(
        id="cloudflare-ai-gateway",
        name="Cloudflare AI Gateway",
        apis=("anthropic-messages", "openai-completions", "openai-responses"),
        auth_name="Cloudflare API key",
        env_vars=("CLOUDFLARE_API_KEY",),
    ),
    MixedProviderSpec(
        id="cloudflare-workers-ai",
        name="Cloudflare Workers AI",
        apis=("openai-completions",),
        auth_name="Cloudflare API key",
        env_vars=("CLOUDFLARE_API_KEY",),
    ),
)

CLOUDFLARE_SPECS_BY_ID = {spec.id: spec for spec in CLOUDFLARE_SPECS}
