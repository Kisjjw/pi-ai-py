from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TypeVar

from pi_ai.http.errors import ProviderHTTPError

T = TypeVar("T")

DEFAULT_MAX_RETRY_DELAY_MS = 60_000


def is_retryable_provider_error(error: ProviderHTTPError) -> bool:
    """照搬 OpenAI / Anthropic SDK 的重试判定。"""
    should_retry = error.get_header("x-should-retry")
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False

    if error.status is None:
        return True
    return error.status in (408, 409, 429) or error.status >= 500


def _validate_server_delay_ms(delay_ms: float, max_retry_delay_ms: int | None, message: str) -> float:
    max_delay_ms = DEFAULT_MAX_RETRY_DELAY_MS if max_retry_delay_ms is None else max_retry_delay_ms
    if max_delay_ms > 0 and delay_ms > max_delay_ms:
        raise RuntimeError(
            f"Server requested {math.ceil(delay_ms / 1000)}s retry delay "
            f"(max: {math.ceil(max_delay_ms / 1000)}s). {message}"
        )
    return delay_ms


def get_retry_delay_ms(
    error: ProviderHTTPError,
    retry_index: int,
    max_retry_delay_ms: int | None,
) -> float:
    retry_after_ms = error.get_header("retry-after-ms")
    if retry_after_ms:
        try:
            return _validate_server_delay_ms(float(retry_after_ms), max_retry_delay_ms, str(error))
        except ValueError:
            pass

    retry_after = error.get_header("retry-after")
    if retry_after:
        try:
            delay_ms = float(retry_after) * 1000
        except ValueError:
            try:
                target = parsedate_to_datetime(retry_after)
            except (TypeError, ValueError):
                target = None
            if target is None:
                delay_ms = 0.0
            else:
                if target.tzinfo is None:
                    target = target.replace(tzinfo=UTC)
                delay_ms = (target - datetime.now(UTC)).total_seconds() * 1000
        return _validate_server_delay_ms(delay_ms, max_retry_delay_ms, str(error))

    exponential_delay_ms: float = min(0.5 * 2**retry_index, 8.0) * 1000
    return exponential_delay_ms * (1 - random.random() * 0.25)


async def retry_provider_request(
    request: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 0,
    max_retry_delay_ms: int | None = None,
) -> T:
    """重放请求直到成功或耗尽重试次数。

    默认不重试。退避期间用 asyncio.sleep，因此外层取消可以直接打断。
    """
    retries_remaining = max_retries

    while True:
        try:
            return await request()
        except ProviderHTTPError as error:
            if retries_remaining <= 0 or not is_retryable_provider_error(error):
                raise
            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            delay_ms = get_retry_delay_ms(error, retry_index, max_retry_delay_ms)
            await asyncio.sleep(max(0.0, delay_ms) / 1000)
