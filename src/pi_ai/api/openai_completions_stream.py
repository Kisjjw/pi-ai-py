from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pi_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
)
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
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from pi_ai.utils.json_parse import parse_streaming_json

#: 按优先级排列的思考字段名。取第一个非空的，避免某些端点同时返回
#: reasoning_content 和 reasoning（内容相同）导致重复。
REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


def map_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    """把厂商的 finish_reason 映射成统一的停止原因。"""
    if reason is None:
        return "stop", None
    if reason in ("stop", "end"):
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in ("function_call", "tool_calls"):
        return "toolUse", None
    if reason == "content_filter":
        return "error", "Provider finish_reason: content_filter"
    if reason == "network_error":
        return "error", "Provider finish_reason: network_error"
    return "error", f"Provider finish_reason: {reason}"


def parse_chunk_usage(raw_usage: dict[str, Any], model: Model) -> Usage:
    """解析用量并算出费用。

    cached_tokens 是缓存命中的读取量。不要从中减去写入量，否则符合规范的
    厂商会被少算。
    """
    prompt_tokens = raw_usage.get("prompt_tokens") or 0
    details = raw_usage.get("prompt_tokens_details") or {}
    cache_read = details.get("cached_tokens")
    if cache_read is None:
        cache_read = raw_usage.get("prompt_cache_hit_tokens") or 0
    cache_write = details.get("cache_write_tokens") or 0

    input_tokens = max(0, prompt_tokens - cache_read - cache_write)
    # completion_tokens 已经包含了推理 token。
    output_tokens = raw_usage.get("completion_tokens") or 0
    completion_details = raw_usage.get("completion_tokens_details") or {}

    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        reasoning=completion_details.get("reasoning_tokens") or 0,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
    )
    calculate_cost(model, usage)
    return usage


def _is_encrypted_reasoning_detail(detail: Any) -> bool:
    return (
        isinstance(detail, dict)
        and detail.get("type") == "reasoning.encrypted"
        and isinstance(detail.get("id"), str)
        and detail["id"]
        and isinstance(detail.get("data"), str)
        and detail["data"]
    )


@dataclass
class _ToolCallState:
    """一个工具调用块的流式状态。

    scratch 字段只在解析期间存在，结束时不会写进最终消息。
    """

    block: ToolCall
    partial_args: str | None = ""
    custom_property: str | None = None
    json_buffer: GrammarToolInputJsonBuffer | None = None
    stream_index: int | None = None

    def current_custom_input(self) -> str:
        if self.custom_property is None:
            return ""
        value = self.block.arguments.get(self.custom_property)
        return value if isinstance(value, str) else ""


@dataclass
class CompletionsStreamParser:
    """把 Chat Completions 的流式 chunk 解析成统一事件。

    TS 侧这段逻辑内联在异步闭包里无法单测，这里抽成独立对象，
    喂 chunk 字典就能验证事件序列。
    """

    model: Model
    grammar_tool_input_properties: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output = AssistantMessage(
            api=self.model.api,
            provider=self.model.provider,
            model=self.model.id,
            stop_reason="pending",
        )
        self._text_block: TextContent | None = None
        self._thinking_block: ThinkingContent | None = None
        self._tool_states: list[_ToolCallState] = []
        self._tools_by_stream_index: dict[int, _ToolCallState] = {}
        self._tools_by_id: dict[str, _ToolCallState] = {}
        self._pending_reasoning_details: dict[str, str] = {}
        self.has_finish_reason = False

    # --- 内部工具 ---

    def _content_index(self, block: Any) -> int:
        for index, candidate in enumerate(self.output.content):
            if candidate is block:
                return index
        return -1

    def _state_for_block(self, block: ToolCall) -> _ToolCallState | None:
        for state in self._tool_states:
            if state.block is block:
                return state
        return None

    # --- chunk 处理 ---

    def feed(self, chunk: Any) -> Iterator[AssistantMessageEvent]:
        if not isinstance(chunk, dict):
            return

        if not self.output.response_id and chunk.get("id"):
            self.output.response_id = chunk["id"]

        chunk_model = chunk.get("model")
        if isinstance(chunk_model, str) and chunk_model and chunk_model != self.model.id:
            if not self.output.response_model:
                self.output.response_model = chunk_model

        if chunk.get("usage"):
            self.output.usage = parse_chunk_usage(chunk["usage"], self.model)

        choices = chunk.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not choice:
            return

        # 兜底：部分厂商（如 Moonshot）把用量放在 choice.usage 而不是 chunk.usage
        if not chunk.get("usage") and choice.get("usage"):
            self.output.usage = parse_chunk_usage(choice["usage"], self.model)

        if choice.get("finish_reason"):
            self.output.raw_stop_reason = choice["finish_reason"]
            stop_reason, error_message = map_stop_reason(choice["finish_reason"])
            self.output.stop_reason = stop_reason
            if error_message:
                self.output.error_message = error_message
            self.has_finish_reason = True

        delta = choice.get("delta")
        if not delta:
            return

        yield from self._handle_text(delta)
        yield from self._handle_thinking(delta)
        yield from self._handle_tool_calls(delta)
        self._handle_reasoning_details(delta)

    def _handle_text(self, delta: dict[str, Any]) -> Iterator[AssistantMessageEvent]:
        content = delta.get("content")
        if not content:
            return

        if self._text_block is None:
            self._text_block = TextContent(text="")
            self.output.content.append(self._text_block)
            yield TextStartEvent(content_index=self._content_index(self._text_block), partial=self.output)

        self._text_block.text += content
        yield TextDeltaEvent(
            content_index=self._content_index(self._text_block), delta=content, partial=self.output
        )

    def _handle_thinking(self, delta: dict[str, Any]) -> Iterator[AssistantMessageEvent]:
        found_field: str | None = None
        for name in REASONING_FIELDS:
            value = delta.get(name)
            if isinstance(value, str) and value:
                found_field = name
                break
        if found_field is None:
            return

        text = delta[found_field]
        signature = found_field
        if self.model.provider == "opencode-go" and found_field == "reasoning":
            signature = "reasoning_content"

        if self._thinking_block is None:
            self._thinking_block = ThinkingContent(thinking="", thinking_signature=signature)
            self.output.content.append(self._thinking_block)
            yield ThinkingStartEvent(
                content_index=self._content_index(self._thinking_block), partial=self.output
            )

        self._thinking_block.thinking += text
        yield ThinkingDeltaEvent(
            content_index=self._content_index(self._thinking_block), delta=text, partial=self.output
        )

    def _ensure_tool_state(self, raw: dict[str, Any]) -> tuple[_ToolCallState, bool]:
        """按 index 优先、id 其次归并工具调用块。返回 (状态, 是否新建)。"""
        stream_index = raw.get("index") if isinstance(raw.get("index"), int) else None
        name = (raw.get("function") or {}).get("name") or (raw.get("custom") or {}).get("name") or ""

        state: _ToolCallState | None = None
        if stream_index is not None:
            state = self._tools_by_stream_index.get(stream_index)
        if state is None and raw.get("id"):
            state = self._tools_by_id.get(raw["id"])

        created = False
        if state is None:
            # 语法工具的输入属性名从工具定义推断；模型编造未知工具时退回 "input"，
            # 至少有个地方存放内容。
            is_custom = bool(raw.get("custom")) and not raw.get("function")
            custom_property = (
                (self.grammar_tool_input_properties.get(name) or "input") if is_custom else None
            )
            block = ToolCall(
                id=raw.get("id") or "",
                name=name,
                arguments={custom_property: ""} if custom_property else {},
            )
            state = _ToolCallState(
                block=block,
                partial_args=None if custom_property else "",
                custom_property=custom_property,
                json_buffer=GrammarToolInputJsonBuffer() if custom_property else None,
                stream_index=stream_index,
            )
            self._tool_states.append(state)
            self.output.content.append(block)
            if stream_index is not None:
                self._tools_by_stream_index[stream_index] = state
            if raw.get("id"):
                self._tools_by_id[raw["id"]] = state
            created = True

        if stream_index is not None and state.stream_index is None:
            state.stream_index = stream_index
            self._tools_by_stream_index[stream_index] = state
        if raw.get("id"):
            self._tools_by_id[raw["id"]] = state
        if not state.block.name and name:
            state.block.name = name

        if raw.get("custom") and not raw.get("function") and state.custom_property is None:
            custom_property = self.grammar_tool_input_properties.get(state.block.name) or "input"
            state.block.arguments = {custom_property: ""}
            state.custom_property = custom_property
            state.json_buffer = GrammarToolInputJsonBuffer()
            state.partial_args = None

        self._apply_pending_reasoning_detail(state)
        return state, created

    def _apply_pending_reasoning_detail(self, state: _ToolCallState) -> None:
        if not state.block.id:
            return
        pending = self._pending_reasoning_details.pop(state.block.id, None)
        if pending:
            state.block.thought_signature = pending

    def _append_custom_input(self, state: _ToolCallState, next_input: str, close: bool) -> str | None:
        if state.custom_property is None or state.json_buffer is None:
            return None
        delta = append_grammar_tool_input_json_delta(
            state.json_buffer, state.custom_property, next_input, close
        )
        state.block.arguments = {state.custom_property: next_input}
        return delta

    def _handle_tool_calls(self, delta: dict[str, Any]) -> Iterator[AssistantMessageEvent]:
        tool_calls = delta.get("tool_calls")
        if not tool_calls:
            return

        for raw in tool_calls:
            state, created = self._ensure_tool_state(raw)
            if created:
                yield ToolCallStartEvent(
                    content_index=self._content_index(state.block), partial=self.output
                )

            if not state.block.id and raw.get("id"):
                state.block.id = raw["id"]
                self._tools_by_id[raw["id"]] = state
            name = (raw.get("function") or {}).get("name") or (raw.get("custom") or {}).get("name")
            if not state.block.name and name:
                state.block.name = name

            emitted = ""
            function_args = (raw.get("function") or {}).get("arguments")
            custom_input = (raw.get("custom") or {}).get("input")
            if function_args:
                emitted = function_args
                state.partial_args = (state.partial_args or "") + function_args
                state.block.arguments = parse_streaming_json(state.partial_args)
            elif custom_input:
                next_input = state.current_custom_input() + custom_input
                emitted = self._append_custom_input(state, next_input, False) or ""

            yield ToolCallDeltaEvent(
                content_index=self._content_index(state.block), delta=emitted, partial=self.output
            )

    def _handle_reasoning_details(self, delta: dict[str, Any]) -> None:
        details = delta.get("reasoning_details")
        if not isinstance(details, list):
            return
        for detail in details:
            if not _is_encrypted_reasoning_detail(detail):
                continue
            serialized = json.dumps(detail, ensure_ascii=False, separators=(",", ":"))
            state = self._tools_by_id.get(detail["id"])
            if state is not None:
                state.block.thought_signature = serialized
            else:
                self._pending_reasoning_details[detail["id"]] = serialized

    # --- 收尾 ---

    def finish(self) -> Iterator[AssistantMessageEvent]:
        """按 content 顺序结束每个块，并清掉解析期的临时状态。"""
        for block in list(self.output.content):
            content_index = self._content_index(block)
            if content_index == -1:
                continue

            if block.type == "text":
                yield TextEndEvent(content_index=content_index, content=block.text, partial=self.output)
            elif block.type == "thinking":
                yield ThinkingEndEvent(
                    content_index=content_index, content=block.thinking, partial=self.output
                )
            elif block.type == "toolCall":
                state = self._state_for_block(block)
                if state is not None:
                    if state.custom_property is not None:
                        delta = self._append_custom_input(state, state.current_custom_input(), True)
                        if delta is not None:
                            yield ToolCallDeltaEvent(
                                content_index=content_index, delta=delta, partial=self.output
                            )
                    else:
                        block.arguments = parse_streaming_json(state.partial_args)
                yield ToolCallEndEvent(
                    content_index=content_index, tool_call=block, partial=self.output
                )

        self._tool_states = []
        self._tools_by_stream_index = {}
        self._tools_by_id = {}

    def infer_missing_stop_reason(self) -> None:
        """厂商不返回 finish_reason 时，按有无工具调用推断。"""
        self.output.stop_reason = (
            "toolUse" if any(b.type == "toolCall" for b in self.output.content) else "stop"
        )
