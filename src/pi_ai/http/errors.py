from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_PROVIDER_ERROR_BODY_CHARS = 4000


class ProviderHTTPError(Exception):
    """厂商 HTTP 错误。

    替代 TS 侧 SDK 异常的形状：重试判定读 status 与 headers，
    错误信息归一化读 body 与 error。
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        headers: Mapping[str, str] | None = None,
        body: str | None = None,
        error: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.headers: dict[str, str] = dict(headers or {})
        self.body = body
        self.error = error
        self._lower_headers = {key.lower(): value for key, value in self.headers.items()}

    def get_header(self, name: str) -> str | None:
        return self._lower_headers.get(name.lower())


@dataclass(frozen=True)
class NormalizedProviderError:
    message: str
    message_carries_body: bool
    status: int | None = None
    body: str | None = None


def safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def truncate_error_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def _extract_body(error: ProviderHTTPError) -> str | None:
    if isinstance(error.body, str):
        body_text: str | None = error.body
    elif isinstance(error.error, dict) and error.error:
        body_text = json.dumps(error.error, ensure_ascii=False)
    else:
        body_text = None

    if body_text is None:
        return None
    trimmed = body_text.strip()
    if not trimmed:
        return None
    return truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS)


def normalize_provider_error(error: BaseException) -> NormalizedProviderError:
    if not isinstance(error, ProviderHTTPError):
        return NormalizedProviderError(message=str(error), message_carries_body=True)

    body = _extract_body(error)
    message = str(error)
    return NormalizedProviderError(
        message=message,
        message_carries_body=body is None or body in message,
        status=error.status,
        body=body,
    )


def format_provider_error(norm: NormalizedProviderError, prefix: str | None = None) -> str:
    """把归一化后的错误拼成展示字符串。

    无前缀：'<status>: <body>'；有前缀：'<prefix> (<status>): <body>'。
    message 已含 body 或缺 status/body 时直接返回 message。
    """
    if norm.message_carries_body or norm.status is None or norm.body is None:
        if prefix is not None and norm.status is not None:
            return f"{prefix} ({norm.status}): {norm.message}"
        return norm.message
    if prefix is not None:
        return f"{prefix} ({norm.status}): {norm.body}"
    return f"{norm.status}: {norm.body}"
