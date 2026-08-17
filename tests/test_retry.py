import asyncio

import pytest

from pi_ai.http.errors import ProviderHTTPError
from pi_ai.http.retry import (
    get_retry_delay_ms,
    is_retryable_provider_error,
    retry_provider_request,
)


def error(status=None, headers=None):
    return ProviderHTTPError("boom", status=status, headers=headers or {})


def test_x_should_retry_true_forces_retry_even_on_400():
    assert is_retryable_provider_error(error(400, {"x-should-retry": "true"})) is True


def test_x_should_retry_false_blocks_retry_even_on_500():
    assert is_retryable_provider_error(error(500, {"x-should-retry": "false"})) is False


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_retryable_statuses(status):
    assert is_retryable_provider_error(error(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_statuses(status):
    assert is_retryable_provider_error(error(status)) is False


def test_missing_status_is_retryable():
    assert is_retryable_provider_error(error(None)) is True


def test_retry_after_ms_header_wins():
    assert get_retry_delay_ms(error(429, {"retry-after-ms": "1500"}), 0, None) == 1500


def test_retry_after_seconds_header_is_converted():
    assert get_retry_delay_ms(error(429, {"retry-after": "2"}), 0, None) == 2000


def test_server_delay_above_cap_raises():
    with pytest.raises(RuntimeError, match="retry delay"):
        get_retry_delay_ms(error(429, {"retry-after": "120"}), 0, None)


def test_cap_of_zero_disables_the_limit():
    assert get_retry_delay_ms(error(429, {"retry-after": "120"}), 0, 0) == 120000


def test_exponential_backoff_is_jittered_within_bounds():
    for retry_index in range(5):
        delay = get_retry_delay_ms(error(500), retry_index, None)
        ceiling = min(0.5 * 2**retry_index, 8) * 1000
        assert ceiling * 0.75 <= delay <= ceiling


async def test_returns_result_without_retrying_on_success():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        return "ok"

    assert await retry_provider_request(request) == "ok"
    assert calls == 1


async def test_does_not_retry_by_default():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise error(500)

    with pytest.raises(ProviderHTTPError):
        await retry_provider_request(request)
    assert calls == 1


async def test_retries_up_to_max_retries():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise error(500, {"retry-after-ms": "0"})

    with pytest.raises(ProviderHTTPError):
        await retry_provider_request(request, max_retries=2)
    assert calls == 3


async def test_stops_retrying_once_request_succeeds():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise error(500, {"retry-after-ms": "0"})
        return "ok"

    assert await retry_provider_request(request, max_retries=3) == "ok"
    assert calls == 2


async def test_does_not_retry_non_provider_errors():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await retry_provider_request(request, max_retries=3)
    assert calls == 1


async def test_backoff_sleep_is_cancellable():
    async def request():
        raise error(500, {"retry-after-ms": "5000"})

    task = asyncio.create_task(retry_provider_request(request, max_retries=1))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
