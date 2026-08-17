from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.types import Context, Model


class ProviderStreams(Protocol):
    def stream(self, model: Model, context: Context, options: Mapping[str, Any]) -> AssistantMessageEventStream: ...

    def stream_simple(
        self, model: Model, context: Context, options: Mapping[str, Any]
    ) -> AssistantMessageEventStream: ...


@dataclass
class Provider:
    """一个厂商：自己的模型目录、认证方式、以及所用的协议实现。

    api 为单个实现时所有模型共用；为映射时按 model.api 分派，
    支持 OpenCode、GitHub Copilot 这类混合协议厂商。
    """

    id: str
    name: str
    base_url: str
    auth: dict[str, Any]
    api: ProviderStreams | dict[str, ProviderStreams]
    models: list[Model] = field(default_factory=list)
    fetch_models: Callable[[], Awaitable[Sequence[Model]]] | None = None

    def get_models(self) -> list[Model]:
        return list(self.models)

    def streams_for(self, model: Model) -> ProviderStreams:
        if isinstance(self.api, dict):
            streams = self.api.get(model.api)
            if streams is None:
                raise LookupError(f"provider {self.id!r} has no implementation for api {model.api!r}")
            return streams
        return self.api

    async def refresh_models(self) -> None:
        if self.fetch_models is None:
            return
        self.models = list(await self.fetch_models())


def create_provider(
    *,
    id: str,
    name: str | None = None,
    base_url: str,
    auth: dict[str, Any],
    api: ProviderStreams | dict[str, ProviderStreams],
    models: Sequence[Model] | None = None,
    fetch_models: Callable[[], Awaitable[Sequence[Model]]] | None = None,
) -> Provider:
    return Provider(
        id=id,
        name=name or id,
        base_url=base_url,
        auth=auth,
        api=api,
        models=list(models or []),
        fetch_models=fetch_models,
    )
