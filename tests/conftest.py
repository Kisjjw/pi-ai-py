"""测试共用工具。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx


class GatedByteStream(httpx.AsyncByteStream):
    """分段吐出响应体，每段之间等一个闸门。

    用来断言「事件在上游还没吐完时就已经交付给消费者」：只要在闸门打开前
    就能读到事件，就说明适配器是边解析边推送，而不是把整条流攒完再推。
    """

    def __init__(self, segments: Sequence[bytes]) -> None:
        if len(segments) < 2:
            raise ValueError("至少要两段才有闸门可言")
        self._segments = list(segments)
        self.gates = [asyncio.Event() for _ in self._segments[1:]]
        self.yielded = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, segment in enumerate(self._segments):
            if index > 0:
                await self.gates[index - 1].wait()
            self.yielded += 1
            yield segment

    def open(self, index: int = 0) -> None:
        self.gates[index].set()

    def open_all(self) -> None:
        for gate in self.gates:
            gate.set()


def gated_response(*segments: str) -> tuple[httpx.Response, GatedByteStream]:
    """构造一个分段闸门化的 SSE 响应，返回 (响应, 闸门流)。"""
    body = GatedByteStream([segment.encode() for segment in segments])
    response = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body
    )
    return response, body


async def read_until(iterator: Any, event_type: str, *, timeout: float = 2.0) -> list[Any]:
    """从事件迭代器读到指定类型为止，返回这一段事件。

    用 wait_for 兜住：若实现退化成「攒完才推」，这里会因为上游卡在闸门上
    而超时失败，而不是永久挂住。
    """
    seen: list[Any] = []
    while True:
        event = await asyncio.wait_for(iterator.__anext__(), timeout)
        seen.append(event)
        if event.type == event_type:
            return seen


async def drain(iterator: Any, *, timeout: float = 2.0) -> list[Any]:
    """把剩余事件读完。"""
    seen: list[Any] = []
    while True:
        try:
            seen.append(await asyncio.wait_for(iterator.__anext__(), timeout))
        except StopAsyncIteration:
            return seen
