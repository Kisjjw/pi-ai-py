from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pi_ai.auth.env import DefaultAuthContext
from pi_ai.auth.store import InMemoryCredentialStore
from pi_ai.auth.types import ApiKeyAuth, ApiKeyCredential, AuthContext, AuthResult, CredentialStore
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.http.client import merge_headers
from pi_ai.provider import Provider
from pi_ai.types import AssistantMessage, Context, ErrorEvent, Model

TransformHeaders = Callable[[dict[str, str]], Awaitable[Mapping[str, str | None]]]


class Models:
    """provider 集合：注册、查询、认证解析，并把请求路由到拥有该模型的 provider。"""

    def __init__(
        self,
        credentials: CredentialStore | None = None,
        auth_context: AuthContext | None = None,
    ) -> None:
        self._providers: dict[str, Provider] = {}
        self._credentials: CredentialStore = credentials or InMemoryCredentialStore()
        self._auth_context: AuthContext = auth_context or DefaultAuthContext()

    # --- 注册与查询 ---

    def set_provider(self, provider: Provider) -> None:
        self._providers[provider.id] = provider

    def get_provider(self, provider_id: str) -> Provider | None:
        return self._providers.get(provider_id)

    def get_providers(self) -> list[Provider]:
        return list(self._providers.values())

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        if provider_id is not None:
            provider = self._providers.get(provider_id)
            return provider.get_models() if provider else []
        return [model for provider in self._providers.values() for model in provider.get_models()]

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        for model in self.get_models(provider_id):
            if model.id == model_id:
                return model
        return None

    async def refresh(self, provider_ids: list[str] | None = None) -> None:
        targets = [
            provider for provider in self._providers.values() if provider_ids is None or provider.id in provider_ids
        ]
        await asyncio.gather(*(provider.refresh_models() for provider in targets))

    # --- 认证 ---

    async def get_auth(self, target: str | Model) -> AuthResult | None:
        """解析 provider 的认证。传 Model 时额外并入 model.headers。"""
        provider_id = target if isinstance(target, str) else target.provider
        provider = self._providers.get(provider_id)
        if provider is None:
            return None

        api_key_auth: ApiKeyAuth | None = provider.auth.get("api_key")
        if api_key_auth is None:
            return None

        stored = await self._credentials.read(provider_id)
        credential = stored if isinstance(stored, ApiKeyCredential) else None
        result: AuthResult | None = await api_key_auth.resolve(ctx=self._auth_context, credential=credential)
        if result is None:
            return None

        if isinstance(target, Model) and target.headers:
            merged = merge_headers(result.auth.headers, target.headers)
            result = result.model_copy(update={"auth": result.auth.model_copy(update={"headers": merged})})
        return result

    # --- 请求 ---

    def stream(self, model: Model, context: Context, **options: Any) -> AssistantMessageEventStream:
        return self._dispatch("stream", model, context, options)

    def stream_simple(self, model: Model, context: Context, **options: Any) -> AssistantMessageEventStream:
        return self._dispatch("stream_simple", model, context, options)

    async def complete(self, model: Model, context: Context, **options: Any) -> AssistantMessage:
        return await self.stream(model, context, **options).result()

    async def complete_simple(self, model: Model, context: Context, **options: Any) -> AssistantMessage:
        return await self.stream_simple(model, context, **options).result()

    def _dispatch(
        self,
        method: str,
        model: Model,
        context: Context,
        options: dict[str, Any],
    ) -> AssistantMessageEventStream:
        stream = AssistantMessageEventStream()
        # 适配器的 stream 要等 _prepare_options 解完认证才建得出来，取消可能来得更早，
        # 所以用一个可变容器持有它，让取消回调随后也能找到。
        inner_holder: list[AssistantMessageEventStream] = []

        def cancel_inner() -> None:
            for inner in inner_holder:
                inner.cancel()

        stream.on_cancel(cancel_inner)

        def error_event(message: str) -> ErrorEvent:
            return ErrorEvent(
                reason="error",
                error=AssistantMessage(
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    stop_reason="error",
                    error_message=message,
                ),
            )

        async def run() -> None:
            try:
                provider = self._providers.get(model.provider)
                if provider is None:
                    raise LookupError(f"no provider registered for {model.provider!r}")

                prepared = await self._prepare_options(provider, model, options)
                streams = provider.streams_for(model)
                inner = getattr(streams, method)(model, context, prepared)
            except Exception as error:  # noqa: BLE001 - 契约要求不抛出
                stream.push(error_event(str(error)))
                return

            inner_holder.append(inner)
            if stream.cancelled:
                inner.cancel()

            # relay 而非 push：事件出自 inner，partial 已经冻结过了。这里继续把
            # 事件转下去（包括 inner 因取消而产出的 aborted 事件），外层才能
            # settle 出正确的最终消息。
            async for event in inner:
                stream.relay(event)
            if not stream.done:
                try:
                    stream.end(await inner.result())
                except Exception as error:  # noqa: BLE001 - 契约要求不抛出
                    stream.push(error_event(str(error)))

        task = asyncio.ensure_future(run())
        stream.settle_with(task)
        return stream

    async def _prepare_options(
        self,
        provider: Provider,
        model: Model,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """合并认证、模型 header、调用方 header，并执行 transform_headers。

        顺序：provider 认证 header -> model.headers -> 显式 headers -> transform_headers
        """
        prepared = dict(options)
        transform: TransformHeaders | None = prepared.pop("transform_headers", None)
        explicit_key = prepared.get("api_key")

        auth_headers: Mapping[str, str | None] | None = None
        if not explicit_key:
            resolved = await self.get_auth(provider.id)
            if resolved is None:
                raise RuntimeError(f"provider {provider.id!r} is not configured (no API key)")
            prepared["api_key"] = resolved.auth.api_key
            auth_headers = resolved.auth.headers
            if resolved.auth.base_url:
                prepared.setdefault("base_url", resolved.auth.base_url)
            if resolved.env:
                prepared["env"] = {**resolved.env, **(prepared.get("env") or {})}

        headers = merge_headers(auth_headers, model.headers, prepared.get("headers"))
        if transform is not None:
            headers = merge_headers(await transform(headers))
        prepared["headers"] = headers
        return prepared


def create_models(
    credentials: CredentialStore | None = None,
    auth_context: AuthContext | None = None,
) -> Models:
    return Models(credentials=credentials, auth_context=auth_context)
