from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from pi_ai.http.errors import ProviderHTTPError

DEFAULT_TIMEOUT_MS = 600_000


@dataclass(frozen=True)
class ProviderResponse:
    status: int
    headers: dict[str, str]


def merge_headers(*layers: Mapping[str, str | None] | None) -> dict[str, str]:
    """按层合并请求头，大小写不敏感，后层覆盖前层。

    值为 None 表示删除该 header，用于压掉底层默认值。
    """
    merged: dict[str, tuple[str, str]] = {}
    for layer in layers:
        if not layer:
            continue
        for name, value in layer.items():
            key = name.lower()
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = (name, value)
    return {name: value for name, value in merged.values()}


class OpenProviderStream:
    """已建立连接、响应体尚未读取的流式响应。

    存在的理由是把「建连 + 首个响应」与「读流」拆成两步：前者可以安全重放，
    后者一旦开始就不能重放了。重试只包住前者，读流留给调用方，这样事件才能
    随上游边到边推送。调用方必须用 async with（或显式 aclose）释放连接。
    """

    def __init__(
        self,
        *,
        response: ProviderResponse,
        body: AsyncIterator[bytes],
        raw: httpx.Response,
        owned_client: httpx.AsyncClient | None,
    ) -> None:
        self.response = response
        self.body = body
        self._raw = raw
        self._owned_client = owned_client
        self._closed = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # 用 finally 保证即使关闭响应时被取消打断，自建的 client 也不会泄漏。
        try:
            await self._raw.aclose()
        finally:
            if self._owned_client is not None:
                await self._owned_client.aclose()

    async def __aenter__(self) -> OpenProviderStream:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


async def open_stream_post(
    *,
    url: str,
    headers: Mapping[str, str],
    json_body: Any,
    timeout_ms: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> OpenProviderStream:
    """发起流式 POST，拿到首个响应就返回，响应体交给调用方消费。

    非 2xx 时读完响应体再抛 ProviderHTTPError，让重试和错误展示拿到完整信息。
    传输层异常同样包装成 ProviderHTTPError（status 为 None，按可重试处理）。
    任何失败路径都会在抛出前释放已占用的连接。
    """
    timeout_seconds = (DEFAULT_TIMEOUT_MS if timeout_ms is None else timeout_ms) / 1000
    owns_client = client is None
    active = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds), trust_env=True)

    try:
        request = active.build_request("POST", url, headers=dict(headers), json=json_body)
        try:
            response = await active.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise ProviderHTTPError(str(exc) or exc.__class__.__name__) from exc

        try:
            if response.status_code >= 400:
                raw = await response.aread()
                body = raw.decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(body)
                except ValueError:
                    parsed = None
                raise ProviderHTTPError(
                    f"{response.status_code} {body}" if body else f"{response.status_code}",
                    status=response.status_code,
                    headers=dict(response.headers),
                    body=body or None,
                    error=parsed,
                )
        except BaseException:
            await response.aclose()
            raise

        return OpenProviderStream(
            response=ProviderResponse(status=response.status_code, headers=dict(response.headers)),
            body=response.aiter_bytes(),
            raw=response,
            owned_client=active if owns_client else None,
        )
    except BaseException:
        if owns_client:
            await active.aclose()
        raise


@asynccontextmanager
async def stream_post(
    *,
    url: str,
    headers: Mapping[str, str],
    json_body: Any,
    timeout_ms: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[tuple[ProviderResponse, AsyncIterator[bytes]]]:
    """open_stream_post 的上下文管理器形态，用于一次性读完整条流的场景。"""
    opened = await open_stream_post(
        url=url, headers=headers, json_body=json_body, timeout_ms=timeout_ms, client=client
    )
    async with opened:
        yield opened.response, opened.body
