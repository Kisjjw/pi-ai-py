from __future__ import annotations

from dataclasses import dataclass

from pi_ai.auth.types import ApiKeyCredential, AuthContext, AuthResult, ModelAuth

ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


@dataclass(frozen=True)
class AnthropicProviderSpec:
    id: str
    name: str
    base_url: str
    auth_name: str
    env_vars: tuple[str, ...]


#: 走 anthropic-messages 协议且认证方式为标准环境变量的厂商。
#: Anthropic 自己的解析顺序更复杂，单独实现，见 AnthropicApiKeyAuth。
ANTHROPIC_COMPATIBLE_SPECS: tuple[AnthropicProviderSpec, ...] = (
    AnthropicProviderSpec(
        id="kimi-coding",
        name="Kimi For Coding",
        base_url="https://api.kimi.com/coding",
        auth_name="Kimi API key",
        env_vars=("KIMI_API_KEY",),
    ),
    AnthropicProviderSpec(
        id="minimax",
        name="MiniMax",
        base_url="https://api.minimax.io/anthropic",
        auth_name="MiniMax API key",
        env_vars=("MINIMAX_API_KEY",),
    ),
    AnthropicProviderSpec(
        id="minimax-cn",
        name="MiniMax CN",
        base_url="https://api.minimaxi.com/anthropic",
        auth_name="MiniMax CN API key",
        env_vars=("MINIMAX_CN_API_KEY",),
    ),
    AnthropicProviderSpec(
        id="vercel-ai-gateway",
        name="Vercel AI Gateway",
        base_url="https://ai-gateway.vercel.sh",
        auth_name="Vercel AI Gateway API key",
        env_vars=("AI_GATEWAY_API_KEY",),
    ),
)

ANTHROPIC_SPEC = AnthropicProviderSpec(
    id="anthropic",
    name="Anthropic",
    base_url="https://api.anthropic.com",
    auth_name="Anthropic API key",
    env_vars=(ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV),
)


class AnthropicApiKeyAuth:
    """Anthropic 的认证解析，比标准的环境变量方式多两条分支。

    顺序：已存凭据 -> ANTHROPIC_AUTH_TOKEN -> ANTHROPIC_OAUTH_TOKEN -> ANTHROPIC_API_KEY。

    注意 ANTHROPIC_AUTH_TOKEN 走的是 **header 通道**（Authorization: Bearer），
    不是 api_key。适配器因此会认为"调用方已提供鉴权"而不再拼 x-api-key。
    """

    name = "Anthropic API key"

    async def resolve(
        self,
        *,
        ctx: AuthContext,
        credential: ApiKeyCredential | None,
    ) -> AuthResult | None:
        if credential is not None and credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=credential.key),
                env=credential.env,
                source="stored credential",
            )

        auth_token = await ctx.env(ANTHROPIC_AUTH_TOKEN_ENV)
        if auth_token:
            return AuthResult(
                auth=ModelAuth(headers={"Authorization": f"Bearer {auth_token}"}),
                source=ANTHROPIC_AUTH_TOKEN_ENV,
            )

        for env_var in (ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV):
            api_key = await ctx.env(env_var)
            if api_key:
                return AuthResult(auth=ModelAuth(api_key=api_key), source=env_var)

        return None
