import httpx
import pytest
import respx

from pi_ai.http.client import merge_headers, open_stream_post, stream_post
from pi_ai.http.errors import ProviderHTTPError


def test_merge_headers_later_layer_wins():
    merged = merge_headers({"a": "1", "b": "2"}, {"b": "3"})
    assert merged == {"a": "1", "b": "3"}


def test_merge_headers_is_case_insensitive():
    merged = merge_headers({"Content-Type": "json"}, {"content-type": "text"})
    assert list(merged.values()) == ["text"]
    assert len(merged) == 1


def test_merge_headers_none_value_deletes():
    merged = merge_headers({"a": "1", "b": "2"}, {"a": None})
    assert merged == {"b": "2"}


def test_merge_headers_skips_none_layers():
    assert merge_headers(None, {"a": "1"}, None) == {"a": "1"}


@respx.mock
async def test_streams_response_body_chunks():
    respx.post("https://x.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, text="data: 1\n\ndata: 2\n\n")
    )

    chunks = []
    async with stream_post(
        url="https://x.test/v1/chat/completions",
        headers={"authorization": "Bearer k"},
        json_body={"model": "m"},
    ) as (response, body):
        assert response.status == 200
        async for chunk in body:
            chunks.append(chunk)

    assert b"".join(chunks) == b"data: 1\n\ndata: 2\n\n"


@respx.mock
async def test_raises_provider_error_with_body_on_non_2xx():
    respx.post("https://x.test/v1/chat/completions").mock(
        return_value=httpx.Response(429, text='{"error":{"message":"slow down"}}', headers={"retry-after": "3"})
    )

    with pytest.raises(ProviderHTTPError) as excinfo:
        async with stream_post(
            url="https://x.test/v1/chat/completions",
            headers={},
            json_body={},
        ) as (_response, _body):
            pass

    error = excinfo.value
    assert error.status == 429
    assert error.get_header("retry-after") == "3"
    assert error.body == '{"error":{"message":"slow down"}}'
    assert error.error == {"error": {"message": "slow down"}}


@respx.mock
async def test_wraps_transport_errors_as_provider_error():
    respx.post("https://x.test/v1/chat/completions").mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(ProviderHTTPError) as excinfo:
        async with stream_post(url="https://x.test/v1/chat/completions", headers={}, json_body={}) as (_r, _b):
            pass

    assert excinfo.value.status is None


# --- open_stream_post：建连与读流分离 ---


@respx.mock
async def test_open_returns_before_the_body_is_read():
    """建连与读流分开，重试才能只包住前者。"""
    read = False

    class TrackingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal read
            read = True
            yield b"data: 1\n\n"

    respx.post("https://x.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, stream=TrackingStream())
    )

    opened = await open_stream_post(
        url="https://x.test/v1/chat/completions", headers={}, json_body={}
    )

    assert opened.response.status == 200
    assert read is False  # 响应体还没被碰过

    async with opened:
        assert b"".join([chunk async for chunk in opened.body]) == b"data: 1\n\n"
    assert read is True


@respx.mock
async def test_open_releases_the_connection_on_error():
    respx.post("https://x.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )

    with pytest.raises(ProviderHTTPError) as excinfo:
        await open_stream_post(url="https://x.test/v1/chat/completions", headers={}, json_body={})

    assert excinfo.value.status == 500


@respx.mock
async def test_aclose_is_idempotent():
    respx.post("https://x.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, text="data: 1\n\n")
    )

    opened = await open_stream_post(
        url="https://x.test/v1/chat/completions", headers={}, json_body={}
    )
    await opened.aclose()
    await opened.aclose()
