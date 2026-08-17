from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pi_ai.types import AnthropicMessagesCompat, Model

FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"

#: 第二段长度达到这个值时说明它是日期戳（claude-opus-4-20250514）而不是次版本号。
_DATE_SUFFIX_MIN_LENGTH = 8

_CLAUDE_VERSION = re.compile(r"^claude-(?:opus|sonnet|fable)-(\d+)(?:-(\d+))?(?:-|$)")


@dataclass(frozen=True)
class AnthropicResolvedCompat:
    supports_eager_tool_input_streaming: bool
    supports_long_cache_retention: bool
    send_session_affinity_headers: bool
    supports_cache_control_on_tools: bool
    supports_temperature: bool
    allow_empty_signature: bool
    supports_strict_tools: bool
    supports_tool_references: bool


def default_supports_tool_references(model: Model) -> bool:
    """`supports_tool_references` 的默认值。

    仅第一方 Anthropic 模型；排除 Haiku（会拒绝客户端发来的 tool_reference 块）
    以及早于工具搜索的版本（Claude 3.x、Opus/Sonnet 4.0、Opus 4.1）。
    """
    if model.provider != "anthropic" or "haiku" in model.id:
        return False

    match = _CLAUDE_VERSION.match(model.id)
    if match is None:
        return False

    major = int(match.group(1))
    minor_raw = match.group(2)
    minor = int(minor_raw) if minor_raw and len(minor_raw) < _DATE_SUFFIX_MIN_LENGTH else 0
    return major > 4 or (major == 4 and minor >= 5)


def _compat_of(model: Model) -> AnthropicMessagesCompat | None:
    compat = model.compat
    if compat is None:
        return None
    # 构造后直接赋值不会触发 pydantic 校验器，这里补一次解析。
    if isinstance(compat, Mapping):
        return AnthropicMessagesCompat.model_validate(compat)
    return compat if isinstance(compat, AnthropicMessagesCompat) else None


def get_anthropic_compat(model: Model) -> AnthropicResolvedCompat:
    compat = _compat_of(model)

    def pick(name: str, fallback: Any) -> Any:
        if compat is None:
            return fallback
        value = getattr(compat, name, None)
        return fallback if value is None else value

    return AnthropicResolvedCompat(
        supports_eager_tool_input_streaming=pick("supports_eager_tool_input_streaming", True),
        supports_long_cache_retention=pick("supports_long_cache_retention", True),
        send_session_affinity_headers=pick("send_session_affinity_headers", False),
        supports_cache_control_on_tools=pick("supports_cache_control_on_tools", True),
        supports_temperature=pick("supports_temperature", True),
        allow_empty_signature=pick("allow_empty_signature", False),
        supports_strict_tools=pick("supports_strict_tools", False),
        supports_tool_references=pick("supports_tool_references", default_supports_tool_references(model)),
    )


def force_adaptive_thinking(model: Model) -> bool:
    """是否强制使用自适应思考。

    TS 侧每次都直接比较 `=== true`，没有并进 resolved 结构，这里照搬：
    只有显式为 True 才算，未设置或 False 都走预算模式。
    """
    compat = _compat_of(model)
    return bool(compat is not None and compat.force_adaptive_thinking is True)
