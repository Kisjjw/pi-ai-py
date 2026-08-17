import asyncio

import pytest

from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    StartEvent,
    TextContent,
    TextDeltaEvent,
)


def make_message(stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(api="a", provider="p", model="m", stop_reason=stop_reason)


async def test_iterates_pushed_events_then_stops():
    stream = AssistantMessageEventStream()
    partial = make_message("pending")
    stream.push(StartEvent(partial=partial))
    stream.push(TextDeltaEvent(content_index=0, delta="hi", partial=partial))
    stream.push(DoneEvent(reason="stop", message=make_message()))

    types = [event.type async for event in stream]
    assert types == ["start", "text_delta", "done"]


async def test_result_resolves_from_done_event():
    stream = AssistantMessageEventStream()
    final = make_message()
    stream.push(DoneEvent(reason="stop", message=final))
    assert await stream.result() is final


async def test_result_is_awaitable_more_than_once():
    stream = AssistantMessageEventStream()
    final = make_message()
    stream.push(DoneEvent(reason="stop", message=final))
    assert await stream.result() is final
    assert await stream.result() is final


async def test_consumer_waits_for_late_events():
    stream = AssistantMessageEventStream()

    async def produce():
        await asyncio.sleep(0.01)
        stream.push(DoneEvent(reason="stop", message=make_message()))

    task = asyncio.create_task(produce())
    types = [event.type async for event in stream]
    await task
    assert types == ["done"]


async def test_cancel_marks_stream_cancelled_and_ends_iteration():
    stream = AssistantMessageEventStream()
    stream.cancel()
    assert stream.cancelled is True
    types = [event.type async for event in stream]
    assert types == []


async def test_result_raises_when_stream_ends_without_final_message():
    stream = AssistantMessageEventStream()
    stream.end()
    with pytest.raises(RuntimeError):
        await stream.result()


async def test_done_flag_tracks_terminal_events():
    stream = AssistantMessageEventStream()
    assert stream.done is False
    stream.push(DoneEvent(reason="stop", message=make_message()))
    assert stream.done is True


async def test_events_pushed_after_termination_are_ignored():
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason="stop", message=make_message()))
    stream.push(TextDeltaEvent(content_index=0, delta="late", partial=make_message("pending")))
    types = [event.type async for event in stream]
    assert types == ["done"]


# --- partial 快照 ---


async def test_push_freezes_partial_against_later_mutation():
    """生产者在事件入队后还会继续改那条消息，已交付的事件不能跟着变。"""
    stream = AssistantMessageEventStream()
    live = make_message("pending")
    live.content.append(TextContent(text="a"))

    stream.push(TextDeltaEvent(content_index=0, delta="a", partial=live))
    live.content[0].text += "b"
    live.usage.output += 7
    live.usage.cost.total += 1.5
    stream.push(DoneEvent(reason="stop", message=live))

    events = [event async for event in stream]

    assert events[0].partial is not live
    assert events[0].partial.content[0].text == "a"
    assert events[0].partial.usage.output == 0
    assert events[0].partial.usage.cost.total == 0.0
    # 终止事件带的是最终消息本身，不冻结：它与 result() 应当是同一个对象
    assert events[1].message is live
    assert await stream.result() is live


async def test_relay_passes_already_frozen_events_through():
    stream = AssistantMessageEventStream()
    event = TextDeltaEvent(content_index=0, delta="a", partial=make_message("pending"))

    stream.relay(event)
    stream.push(DoneEvent(reason="stop", message=make_message()))

    events = [e async for e in stream]
    assert events[0] is event


# --- 取消传播 ---


async def test_cancel_runs_hooks_and_leaves_settling_to_the_producer():
    stream = AssistantMessageEventStream()
    calls = []
    stream.on_cancel(lambda: calls.append("hook"))

    stream.cancel()

    assert calls == ["hook"]
    assert stream.cancelled is True
    # 生产者还要推一条 aborted 事件，所以此刻不能自行结束
    assert stream.done is False

    stream.push(ErrorEvent(reason="aborted", error=make_message("aborted")))
    assert (await stream.result()).stop_reason == "aborted"


async def test_cancel_is_idempotent():
    stream = AssistantMessageEventStream()
    calls = []
    stream.on_cancel(lambda: calls.append("hook"))

    stream.cancel()
    stream.cancel()

    assert calls == ["hook"]


async def test_cancel_after_completion_does_not_run_hooks():
    stream = AssistantMessageEventStream()
    calls = []
    stream.on_cancel(lambda: calls.append("hook"))
    stream.push(DoneEvent(reason="stop", message=make_message()))

    stream.cancel()

    assert calls == []
    assert stream.cancelled is True


async def test_hook_registered_after_cancel_fires_immediately():
    """取消可能早于生产者创建完成，晚到的回调也必须被通知。"""
    stream = AssistantMessageEventStream()
    stream.cancel()

    calls = []
    stream.on_cancel(lambda: calls.append("hook"))

    assert calls == ["hook"]


async def test_settle_with_ends_a_stream_its_producer_abandoned():
    stream = AssistantMessageEventStream()

    async def producer():
        return None

    task = asyncio.create_task(producer())
    stream.settle_with(task)
    await task
    await asyncio.sleep(0)  # 让 done callback 执行

    assert stream.done is True
    with pytest.raises(RuntimeError):
        await stream.result()


async def test_settle_with_does_not_override_a_real_result():
    stream = AssistantMessageEventStream()
    final = make_message()

    async def producer():
        stream.push(DoneEvent(reason="stop", message=final))

    task = asyncio.create_task(producer())
    stream.settle_with(task)
    await task
    await asyncio.sleep(0)

    assert await stream.result() is final
