from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pi_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    TextContent,
    Tool,
    ToolResultMessage,
    Usage,
    UserMessage,
)

CHARS_PER_TOKEN = 4
ESTIMATED_IMAGE_CHARS = 4800


@dataclass(frozen=True)
class ContextUsageEstimate:
    """上下文用量估算结果。

    tokens 是总量；usage_tokens 来自最近一条仍适用的 assistant 用量；
    trailing_tokens 是该用量之后新增内容的估算值。
    """

    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def calculate_context_tokens(usage: Usage) -> int:
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


def _safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "[unserializable]"


def _content_chars(content: str | Sequence[TextContent | ImageContent]) -> int:
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content:
        chars += len(block.text) if block.type == "text" else ESTIMATED_IMAGE_CHARS
    return chars


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_text_and_image_content_tokens(content: str | Sequence[TextContent | ImageContent]) -> int:
    return math.ceil(_content_chars(content) / CHARS_PER_TOKEN)


def estimate_message_tokens(message: Message) -> int:
    if isinstance(message, UserMessage | ToolResultMessage):
        return estimate_text_and_image_content_tokens(message.content)

    chars = 0
    for block in message.content:
        if block.type == "text":
            chars += len(block.text)
        elif block.type == "thinking":
            chars += len(block.thinking)
        else:
            chars += len(block.name) + len(_safe_json_stringify(block.arguments))
    return math.ceil(chars / CHARS_PER_TOKEN)


def _get_last_assistant_usage_info(messages: Sequence[Message]) -> tuple[Usage, int] | None:
    """找出最近一条其用量仍描述当前前缀的 assistant 消息。

    如果它之后被插入了更新的消息（例如压缩摘要），说明前缀已变，
    该用量不再适用。这就是 latest_prefix_timestamp 比较的意义。
    """
    latest_prefix_timestamp = float("-inf")
    usage_info: tuple[Usage, int] | None = None

    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            applies_to_prefix = message.timestamp >= latest_prefix_timestamp
            if (
                applies_to_prefix
                and message.stop_reason not in ("aborted", "error")
                and calculate_context_tokens(message.usage) > 0
            ):
                usage_info = (message.usage, index)
        latest_prefix_timestamp = max(latest_prefix_timestamp, message.timestamp)

    return usage_info


def _estimate_messages(messages: Sequence[Message]) -> ContextUsageEstimate:
    usage_info = _get_last_assistant_usage_info(messages)
    if usage_info is not None:
        usage, index = usage_info
        usage_tokens = calculate_context_tokens(usage)
        trailing_tokens = sum(estimate_message_tokens(m) for m in messages[index + 1 :])
        return ContextUsageEstimate(
            tokens=usage_tokens + trailing_tokens,
            usage_tokens=usage_tokens,
            trailing_tokens=trailing_tokens,
            last_usage_index=index,
        )

    tokens = sum(estimate_message_tokens(m) for m in messages)
    return ContextUsageEstimate(tokens=tokens, usage_tokens=0, trailing_tokens=tokens, last_usage_index=None)


def _estimate_tools_tokens(tools: Sequence[Tool] | None) -> int:
    if not tools:
        return 0
    return estimate_text_tokens(_safe_json_stringify([t.model_dump(by_alias=True, exclude_none=True) for t in tools]))


def estimate_context_tokens(context: Context | Sequence[Message]) -> ContextUsageEstimate:
    if not isinstance(context, Context):
        return _estimate_messages(context)

    estimate = _estimate_messages(context.messages)

    if estimate.last_usage_index is not None:
        added_names: set[str] = set()
        for message in context.messages[estimate.last_usage_index + 1 :]:
            if isinstance(message, ToolResultMessage):
                added_names.update(message.added_tool_names or [])
        added_tools = [t for t in (context.tools or []) if t.name in added_names]
        added_tool_tokens = _estimate_tools_tokens(added_tools)
        return ContextUsageEstimate(
            tokens=estimate.tokens + added_tool_tokens,
            usage_tokens=estimate.usage_tokens,
            trailing_tokens=estimate.trailing_tokens + added_tool_tokens,
            last_usage_index=estimate.last_usage_index,
        )

    prefix_tokens = (estimate_text_tokens(context.system_prompt) if context.system_prompt else 0) + (
        _estimate_tools_tokens(context.tools)
    )
    return ContextUsageEstimate(
        tokens=estimate.tokens + prefix_tokens,
        usage_tokens=estimate.usage_tokens,
        trailing_tokens=estimate.trailing_tokens + prefix_tokens,
        last_usage_index=estimate.last_usage_index,
    )
