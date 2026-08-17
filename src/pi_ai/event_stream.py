from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, Final

from pi_ai.types import AssistantMessage, AssistantMessageEvent, DoneEvent, ErrorEvent
from pi_ai.utils.snapshot import snapshot_assistant_message

_END: Final = object()


def _freeze(event: AssistantMessageEvent) -> AssistantMessageEvent:
    """冻结增量事件的 partial，使它反映事件产生那一刻的状态。

    终止事件（done / error）带的是最终消息，生产者之后不会再改，原样透传 ——
    这样调用方从事件里拿到的对象与 result() 返回的是同一个。
    """
    if isinstance(event, DoneEvent | ErrorEvent):
        return event
    return event.model_copy(update={"partial": snapshot_assistant_message(event.partial)})


class AssistantMessageEventStream:
    """事件队列，支持异步迭代、取最终消息、取消。

    对应 TS 的 utils/event-stream.ts。终止事件是 done 或 error，
    两者都会解析出最终的 AssistantMessage。

    内部用 asyncio.Queue 与 asyncio.Event，两者都延迟绑定事件循环，
    因此本对象可以在协程外构造。
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AssistantMessageEvent | object] = asyncio.Queue()
        self._settled = asyncio.Event()
        self._done = False
        self._cancelled = False
        self._result: AssistantMessage | None = None
        self._error: BaseException | None = None
        self._cancel_hooks: list[Callable[[], object]] = []

    @property
    def done(self) -> bool:
        return self._done

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    # --- 生产 ---

    def push(self, event: AssistantMessageEvent) -> None:
        """推入一个新产出的事件。

        这里会顺带冻结 partial：队列意味着事件与消费者之间有时间差，而生产者
        在这段时间里还在就地修改那条消息。TS 版事件是同步交付的，不存在这个
        时间差，所以那边共享引用是安全的；移植到 asyncio 后必须冻结。
        """
        self._enqueue(_freeze(event))

    def relay(self, event: AssistantMessageEvent) -> None:
        """转发来自上游 stream 的事件，跳过重复冻结（它已经冻结过了）。"""
        self._enqueue(event)

    def _enqueue(self, event: AssistantMessageEvent) -> None:
        if self._done:
            return

        self._queue.put_nowait(event)

        if isinstance(event, DoneEvent | ErrorEvent):
            final = event.message if isinstance(event, DoneEvent) else event.error
            self._done = True
            self._result = final
            self._queue.put_nowait(_END)
            self._settled.set()

    def end(self, result: AssistantMessage | None = None) -> None:
        if self._done:
            return
        self._done = True
        if result is not None:
            self._result = result
        elif self._result is None:
            self._error = RuntimeError("stream ended without a final message")
        self._queue.put_nowait(_END)
        self._settled.set()

    # --- 取消 ---

    def on_cancel(self, hook: Callable[[], object]) -> None:
        """注册取消回调，让 cancel() 能中断真正在跑的生产者。

        注册过回调时 cancel() 不再自行结束队列，而是把收尾交给生产者：它会
        推出一条 stop_reason == "aborted" 的事件，result() 才拿得到 aborted
        而不是「流结束却没有最终消息」的错误。
        """
        if self._cancelled:
            hook()
            return
        self._cancel_hooks.append(hook)

    def settle_with(self, task: asyncio.Future[Any]) -> None:
        """生产者任务结束后，若流还没 settle 就兜底结束它，避免 result() 永久挂住。"""
        task.add_done_callback(self._on_producer_done)

    def bind_producer(self, task: asyncio.Task[Any]) -> None:
        """绑定生产者任务：cancel() 取消它，它结束后兜底 settle 这条流。

        取消要推迟一个事件循环周期再投递。asyncio 对「还没迈出第一步」的任务
        执行 cancel 时，协程体完全不会执行，于是生产者内部的 CancelledError
        分支也不会跑，那条 aborted 事件就永远不会产出。call_soon 能保证取消
        落在第一步之后——任务的第一步在 create_task 时就已排进就绪队列，而
        事件循环按注册顺序回调——这样取消总能进到生产者自己的处理逻辑里，
        产出带已收增量的 aborted 消息。
        """
        self.on_cancel(lambda: task.get_loop().call_soon(task.cancel))
        self.settle_with(task)

    def _on_producer_done(self, _task: asyncio.Future[Any]) -> None:
        if not self._done:
            self.end(self._result)

    def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        if self._done:
            return

        hooks = self._cancel_hooks
        self._cancel_hooks = []
        for hook in hooks:
            hook()
        # 没有生产者可通知时（纯本地流），只能自己结束。
        if not hooks:
            self.end(self._result)

    # --- 消费 ---

    async def result(self) -> AssistantMessage:
        await self._settled.wait()
        if self._result is not None:
            return self._result
        raise self._error or RuntimeError("stream ended without a final message")

    async def __aiter__(self) -> AsyncIterator[AssistantMessageEvent]:
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            yield item  # type: ignore[misc]
