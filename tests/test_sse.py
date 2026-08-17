from collections.abc import AsyncIterator

from pi_ai.http.sse import SSEEvent, iter_sse


async def feed(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def collect(*chunks: bytes) -> list[SSEEvent]:
    return [event async for event in iter_sse(feed(*chunks))]


async def test_parses_named_events():
    events = await collect(b'event: message_start\ndata: {"a":1}\n\n')
    assert events == [SSEEvent(event="message_start", data='{"a":1}')]


async def test_parses_unnamed_data_only_events():
    events = await collect(b'data: {"a":1}\n\n')
    assert events == [SSEEvent(event=None, data='{"a":1}')]


async def test_handles_crlf_line_endings():
    events = await collect(b"event: x\r\ndata: 1\r\n\r\n")
    assert events == [SSEEvent(event="x", data="1")]


async def test_handles_lone_cr_line_endings():
    events = await collect(b"event: x\rdata: 1\r\r")
    assert events == [SSEEvent(event="x", data="1")]


async def test_reassembles_events_split_across_chunks():
    events = await collect(b"event: mes", b'sage\ndata: {"a"', b':1}\n\n')
    assert events == [SSEEvent(event="message", data='{"a":1}')]


async def test_joins_multiple_data_lines_with_newline():
    events = await collect(b"data: line1\ndata: line2\n\n")
    assert events == [SSEEvent(event=None, data="line1\nline2")]


async def test_ignores_comment_lines():
    events = await collect(b": keep-alive\ndata: 1\n\n")
    assert events == [SSEEvent(event=None, data="1")]


async def test_emits_done_sentinel_as_regular_event():
    events = await collect(b"data: [DONE]\n\n")
    assert events == [SSEEvent(event=None, data="[DONE]")]


async def test_flushes_trailing_event_without_blank_line():
    events = await collect(b"data: 1\n")
    assert events == [SSEEvent(event=None, data="1")]


async def test_tolerates_missing_space_after_colon():
    events = await collect(b"data:1\n\n")
    assert events == [SSEEvent(event=None, data="1")]
