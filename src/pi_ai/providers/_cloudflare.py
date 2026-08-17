from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pi_ai.auth.types import ApiKeyCredential, AuthContext, AuthResult, ModelAuth
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.provider import ProviderStreams
from pi_ai.types import Context, Model

CLOUDFLARE_API_KEY = "CLOUDFLARE_API_KEY"
CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"


async def _resolve_value(
    name: str,
    ctx: AuthContext,
    credential: ApiKeyCredential | None,
) -> str | None:
    """逐字段合并：凭据里有就用凭据的，没有就退回环境变量。

    只带了 API key 的凭据仍然要能从环境变量拿到账号 / 网关 id，所以每个值
    都要单独走这套判断，不能因为凭据存在就整体忽略环境变量。
    """
    if credential is not None:
        from_credential = credential.key if name == CLOUDFLARE_API_KEY else (credential.env or {}).get(name)
        if from_credential is not None:
            return from_credential
    return await ctx.env(name)


async def _resolve_cloudflare_env(
    needs_gateway: bool,
    ctx: AuthContext,
    credential: ApiKeyCredential | None,
) -> tuple[str, dict[str, str], str] | None:
    api_key = await _resolve_value(CLOUDFLARE_API_KEY, ctx, credential)
    account_id = await _resolve_value(CLOUDFLARE_ACCOUNT_ID, ctx, credential)
    gateway_id = await _resolve_value(CLOUDFLARE_GATEWAY_ID, ctx, credential) if needs_gateway else None

    if not api_key or not account_id or (needs_gateway and not gateway_id):
        return None

    env = {CLOUDFLARE_ACCOUNT_ID: account_id}
    if gateway_id:
        env[CLOUDFLARE_GATEWAY_ID] = gateway_id

    source = "stored credential" if credential is not None else CLOUDFLARE_API_KEY
    return api_key, env, source


@dataclass
class CloudflareWorkersAIAuth:
    name: str = "Cloudflare API key"

    async def resolve(
        self,
        *,
        ctx: AuthContext,
        credential: ApiKeyCredential | None,
    ) -> AuthResult | None:
        resolved = await _resolve_cloudflare_env(False, ctx, credential)
        if resolved is None:
            return None
        api_key, env, source = resolved
        return AuthResult(auth=ModelAuth(api_key=api_key), env=env, source=source)


@dataclass
class CloudflareAIGatewayAuth:
    name: str = "Cloudflare API key"

    async def resolve(
        self,
        *,
        ctx: AuthContext,
        credential: ApiKeyCredential | None,
    ) -> AuthResult | None:
        resolved = await _resolve_cloudflare_env(True, ctx, credential)
        if resolved is None:
            return None
        api_key, env, source = resolved
        return AuthResult(
            auth=ModelAuth(
                headers={
                    "cf-aig-authorization": f"Bearer {api_key}",
                    # 显式置空以压掉适配器默认拼的鉴权头：网关只认 cf-aig-authorization，
                    # 多余的 Authorization / x-api-key 会被上游拒绝。
                    "Authorization": None,
                    "x-api-key": None,
                }
            ),
            env=env,
            source=source,
        )


def resolve_cloudflare_model(model: Model, env: Mapping[str, str] | None) -> Model:
    """把 base_url 里的账号 / 网关占位符替换成实际值。

    env 里缺某个键时保留原占位符，而不是替换成空串——后者会拼出一个看似
    合法却指向错误资源的 URL。
    """
    if not env:
        return model

    base_url = model.base_url
    for key in (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_GATEWAY_ID):
        placeholder = "{" + key + "}"
        value = env.get(key)
        if value:
            base_url = base_url.replace(placeholder, value)

    if base_url == model.base_url:
        return model
    return model.model_copy(update={"base_url": base_url})


@dataclass
class CloudflareStreams:
    """包一层协议实现，在分派前把端点占位符替换掉。"""

    inner: ProviderStreams

    def stream(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return self.inner.stream(resolve_cloudflare_model(model, options.get("env")), context, options)

    def stream_simple(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream:
        return self.inner.stream_simple(
            resolve_cloudflare_model(model, options.get("env")), context, options
        )


def cloudflare_streams(streams: ProviderStreams) -> CloudflareStreams:
    return CloudflareStreams(inner=streams)
