from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from pi_ai.api.anthropic_messages_messages import from_claude_code_name
from pi_ai.cost import calculate_cost
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Model,
    StopReason,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from pi_ai.utils.json_parse import parse_streaming_json

REDACTED_THINKING_TEXT = "[Reasoning redacted]"

#: 解析器消费的事件类型，其余一律忽略。
CONSUMED_EVENTS = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    }
)


def map_stop_reason(reason: str, stop_details: dict[str, Any] | None = None) -> tuple[StopReason, str | None]:
    """把 Anthropic 的停止原因映射成统一取值。

    未知取值直接抛异常，与 TS 一致：出现新值时宁可报错也不要静默当成正常结束。
    """
    if reason == "end_turn":
        return "stop", None
    if reason == "max_tokens":
        return "length", None
    if reason == "tool_use":
        return "toolUse", None
    if reason == "refusal":
        explanation = (stop_details or {}).get("explanation")
        return "error", explanation or "The model refused to complete the request"
    if reason == "pause_turn":
        # stop 就够了，调用方会重新提交
        return "stop", None
    if reason == "stop_sequence":
        # 我们不发 stop_sequences，正常不会走到这里
        return "stop", None
    if reason == "sensitive":
        return "error", "Provider stopped with: sensitive"
    raise ValueError(f"Unhandled stop reason: {reason}")


@dataclass
class _BlockState:
    """一个内容块的流式状态。

    anthropic_index 是 Anthropic 事件里的编号，与 output.content 的下标不是一回事。
    """

    block: Any
    anthropic_index: int
    partial_json: str = ""


@dataclass
class AnthropicStreamParser:
    model: Model
    is_oauth: bool = False
    tools: Sequence[Tool] | None = None
    saw_message_start: bool = field(default=False, init=False)
    saw_message_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.output = AssistantMessage(
            api=self.model.api,
            provider=self.model.provider,
            model=self.model.id,
            stop_reason="pending",
        )
        self._blocks: list[_BlockState] = []

    def _find(self, anthropic_index: int) -> tuple[int, _BlockState] | None:
        for content_index, state in enumerate(self._blocks):
            if state.anthropic_index == anthropic_index:
                return content_index, state
        return None

    def feed(self, event: dict[str, Any]) -> Iterator[AssistantMessageEvent]:
        event_type = event.get("type")
        if event_type not in CONSUMED_EVENTS:
            return

        if event_type == "message_start":
            self.saw_message_start = True
            self._handle_message_start(event)
        elif event_type == "message_stop":
            self.saw_message_stop = True
        elif event_type == "content_block_start":
            yield from self._handle_block_start(event)
        elif event_type == "content_block_delta":
            yield from self._handle_block_delta(event)
        elif event_type == "content_block_stop":
            yield from self._handle_block_stop(event)
        elif event_type == "message_delta":
            self._handle_message_delta(event)

    def _handle_message_start(self, event: dict[str, Any]) -> None:
        message = event.get("message") or {}
        self.output.response_id = message.get("id")

        usage = message.get("usage") or {}
        self.output.usage.input = usage.get("input_tokens") or 0
        self.output.usage.output = usage.get("output_tokens") or 0
        self.output.usage.cache_read = usage.get("cache_read_input_tokens") or 0
        self.output.usage.cache_write = usage.get("cache_creation_input_tokens") or 0
        self.output.usage.cache_write_1h = (usage.get("cache_creation") or {}).get(
            "ephemeral_1h_input_tokens"
        ) or 0
        self._recompute_usage()

    def _handle_block_start(self, event: dict[str, Any]) -> Iterator[AssistantMessageEvent]:
        anthropic_index = event.get("index", 0)
        raw = event.get("content_block") or {}
        block_type = raw.get("type")

        if block_type == "text":
            block: Any = TextContent(text=raw.get("text") or "")
            event_cls: Any = TextStartEvent
        elif block_type == "thinking":
            block = ThinkingContent(
                thinking=raw.get("thinking") or "", thinking_signature=raw.get("signature") or ""
            )
            event_cls = ThinkingStartEvent
        elif block_type == "redacted_thinking":
            block = ThinkingContent(
                thinking=REDACTED_THINKING_TEXT,
                thinking_signature=raw.get("data"),
                redacted=True,
            )
            event_cls = ThinkingStartEvent
        elif block_type == "tool_use":
            name = raw.get("name") or ""
            block = ToolCall(
                id=raw.get("id") or "",
                name=from_claude_code_name(name, self.tools) if self.is_oauth else name,
                arguments=raw.get("input") or {},
            )
            event_cls = ToolCallStartEvent
        else:
            return

        self.output.content.append(block)
        self._blocks.append(_BlockState(block=block, anthropic_index=anthropic_index))
        yield event_cls(content_index=len(self.output.content) - 1, partial=self.output)

    def _handle_block_delta(self, event: dict[str, Any]) -> Iterator[AssistantMessageEvent]:
        found = self._find(event.get("index", 0))
        if found is None:
            return
        content_index, state = found
        delta = event.get("delta") or {}
        delta_type = delta.get("type")

        if delta_type == "text_delta" and state.block.type == "text":
            text = delta.get("text") or ""
            state.block.text += text
            yield TextDeltaEvent(content_index=content_index, delta=text, partial=self.output)
        elif delta_type == "thinking_delta" and state.block.type == "thinking":
            text = delta.get("thinking") or ""
            state.block.thinking += text
            yield ThinkingDeltaEvent(content_index=content_index, delta=text, partial=self.output)
        elif delta_type == "input_json_delta" and state.block.type == "toolCall":
            fragment = delta.get("partial_json") or ""
            state.partial_json += fragment
            state.block.arguments = parse_streaming_json(state.partial_json)
            yield ToolCallDeltaEvent(content_index=content_index, delta=fragment, partial=self.output)
        elif delta_type == "signature_delta" and state.block.type == "thinking":
            # 签名只累积，不发事件
            state.block.thinking_signature = (state.block.thinking_signature or "") + (
                delta.get("signature") or ""
            )

    def _handle_block_stop(self, event: dict[str, Any]) -> Iterator[AssistantMessageEvent]:
        found = self._find(event.get("index", 0))
        if found is None:
            return
        content_index, state = found
        block = state.block

        if block.type == "text":
            yield TextEndEvent(content_index=content_index, content=block.text, partial=self.output)
        elif block.type == "thinking":
            yield ThinkingEndEvent(content_index=content_index, content=block.thinking, partial=self.output)
        elif block.type == "toolCall":
            block.arguments = parse_streaming_json(state.partial_json)
            state.partial_json = ""
            yield ToolCallEndEvent(content_index=content_index, tool_call=block, partial=self.output)

    def _handle_message_delta(self, event: dict[str, Any]) -> None:
        delta = event.get("delta") or {}
        if delta.get("stop_reason"):
            self.output.raw_stop_reason = delta["stop_reason"]
            stop_reason, error_message = map_stop_reason(delta["stop_reason"], delta.get("stop_details"))
            self.output.stop_reason = stop_reason
            if error_message:
                self.output.error_message = error_message

        usage = event.get("usage")
        if usage:
            # 只在字段存在时覆盖：某些代理在 message_delta 里省略 input_tokens，
            # 无条件覆盖会把 message_start 拿到的值冲掉。
            if usage.get("input_tokens") is not None:
                self.output.usage.input = usage["input_tokens"]
            if usage.get("output_tokens") is not None:
                self.output.usage.output = usage["output_tokens"]
            if usage.get("cache_read_input_tokens") is not None:
                self.output.usage.cache_read = usage["cache_read_input_tokens"]
            if usage.get("cache_creation_input_tokens") is not None:
                self.output.usage.cache_write = usage["cache_creation_input_tokens"]
            thinking_tokens = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
            if thinking_tokens is not None:
                self.output.usage.reasoning = thinking_tokens

        self._recompute_usage()

    def _recompute_usage(self) -> None:
        """Anthropic 不给 total_tokens，自己按分项加出来。"""
        usage = self.output.usage
        usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write
        calculate_cost(self.model, usage)
