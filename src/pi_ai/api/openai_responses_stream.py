from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from pi_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
)
from pi_ai.api.openai_responses_shared import encode_text_signature_v1
from pi_ai.cost import calculate_cost
from pi_ai.http.errors import ProviderHTTPError
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


def map_stop_reason(status: str | None, incomplete_reason: str | None = None) -> tuple[StopReason, str | None]:
    if not status:
        return "stop", None
    if status == "completed":
        return "stop", None
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return "length", None
        return "error", (
            f"Response incomplete: {incomplete_reason}"
            if incomplete_reason
            else "Response incomplete without a provider reason"
        )
    if status in ("failed", "cancelled"):
        return "error", None
    # 这两个状态出现在流式终态里本身就不太合理，但按 stop 处理不影响调用方
    if status in ("in_progress", "queued"):
        return "stop", None
    raise ValueError(f"Unhandled stop reason: {status}")


@dataclass
class _Slot:
    """一个 output_index 对应的内容块状态。"""

    kind: str  # "thinking" | "text" | "toolCall"
    block: Any
    content_index: int
    partial_json: str | None = None
    custom_property: str | None = None
    json_buffer: GrammarToolInputJsonBuffer | None = None

    def current_custom_input(self) -> str:
        if self.custom_property is None:
            return ""
        value = self.block.arguments.get(self.custom_property)
        return value if isinstance(value, str) else ""


@dataclass
class ResponsesStreamParser:
    """把 Responses 的事件流解析成统一事件。

    与 Chat Completions 的差别：这里的块由 output_index 索引，且每个块的
    生命周期由 output_item.added / output_item.done 明确界定。
    """

    model: Model
    grammar_tool_input_properties: Mapping[str, str] = field(default_factory=dict)
    service_tier: str | None = None
    apply_service_tier_pricing: Callable[[Usage, str | None], None] | None = None
    resolve_service_tier: Callable[[str | None, str | None], str | None] | None = None

    def __post_init__(self) -> None:
        self.output = AssistantMessage(
            api=self.model.api,
            provider=self.model.provider,
            model=self.model.id,
            stop_reason="pending",
        )
        self._slots: dict[int, _Slot] = {}
        self._reasoning_blocks_by_id: dict[str, ThinkingContent] = {}
        self.saw_terminal_event = False

    # --- 槽位 ---

    def _get_slot(self, output_index: int, kind: str) -> _Slot | None:
        slot = self._slots.get(output_index)
        return slot if slot is not None and slot.kind == kind else None

    def _create_slot(self, output_index: int, item: Mapping[str, Any]) -> tuple[_Slot | None, Any]:
        """按 item 类型建槽位，返回 (槽位, 需要发出的 start 事件)。"""
        item_type = item.get("type")

        if item_type == "reasoning":
            block: Any = ThinkingContent(thinking="")
            self.output.content.append(block)
            slot = _Slot(kind="thinking", block=block, content_index=len(self.output.content) - 1)
            self._slots[output_index] = slot
            return slot, ThinkingStartEvent(content_index=slot.content_index, partial=self.output)

        if item_type == "message":
            self._apply_message_phase(item)
            block = TextContent(text="")
            self.output.content.append(block)
            slot = _Slot(kind="text", block=block, content_index=len(self.output.content) - 1)
            self._slots[output_index] = slot
            return slot, TextStartEvent(content_index=slot.content_index, partial=self.output)

        if item_type == "function_call":
            block = ToolCall(
                id=f"{item.get('call_id', '')}|{item.get('id', '')}",
                name=item.get("name") or "",
                arguments={},
                namespace=item.get("namespace"),
            )
            self.output.content.append(block)
            slot = _Slot(
                kind="toolCall",
                block=block,
                content_index=len(self.output.content) - 1,
                partial_json=item.get("arguments") or "",
            )
            self._slots[output_index] = slot
            return slot, ToolCallStartEvent(content_index=slot.content_index, partial=self.output)

        if item_type == "custom_tool_call":
            name = item.get("name") or ""
            input_property = self.grammar_tool_input_properties.get(name) or "input"
            block = ToolCall(
                id=f"{item.get('call_id', '')}|{item.get('id', '')}",
                name=name,
                arguments={input_property: item.get("input") or ""},
                namespace=item.get("namespace"),
            )
            self.output.content.append(block)
            slot = _Slot(
                kind="toolCall",
                block=block,
                content_index=len(self.output.content) - 1,
                custom_property=input_property,
                json_buffer=GrammarToolInputJsonBuffer(),
            )
            self._slots[output_index] = slot
            return slot, ToolCallStartEvent(content_index=slot.content_index, partial=self.output)

        return None, None

    def _append_custom_input(self, slot: _Slot, next_input: str, close: bool) -> str | None:
        if slot.custom_property is None or slot.json_buffer is None:
            return None
        delta = append_grammar_tool_input_json_delta(
            slot.json_buffer, slot.custom_property, next_input, close
        )
        slot.block.arguments = {slot.custom_property: next_input}
        return delta

    def _apply_message_phase(self, item: Mapping[str, Any]) -> None:
        """final_answer 阶段的消息提前把停止原因定为 stop。"""
        if item.get("type") == "message" and item.get("phase") == "final_answer":
            self.output.stop_reason = "stop"

    # --- 事件分发 ---

    def feed(self, event: Mapping[str, Any]) -> Iterator[AssistantMessageEvent]:
        event_type = event.get("type")

        if event_type == "response.created":
            self.output.response_id = (event.get("response") or {}).get("id")

        elif event_type == "response.output_item.added":
            _slot, start_event = self._create_slot(event.get("output_index", 0), event.get("item") or {})
            if start_event is not None:
                yield start_event

        elif event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            slot = self._get_slot(event.get("output_index", 0), "thinking")
            if slot is not None:
                delta = event.get("delta") or ""
                slot.block.thinking += delta
                yield ThinkingDeltaEvent(
                    content_index=slot.content_index, delta=delta, partial=self.output
                )

        elif event_type == "response.reasoning_summary_part.done":
            slot = self._get_slot(event.get("output_index", 0), "thinking")
            if slot is not None:
                slot.block.thinking += "\n\n"
                yield ThinkingDeltaEvent(
                    content_index=slot.content_index, delta="\n\n", partial=self.output
                )

        elif event_type in ("response.output_text.delta", "response.refusal.delta"):
            slot = self._get_slot(event.get("output_index", 0), "text")
            if slot is not None:
                delta = event.get("delta") or ""
                slot.block.text += delta
                yield TextDeltaEvent(content_index=slot.content_index, delta=delta, partial=self.output)

        elif event_type == "response.function_call_arguments.delta":
            slot = self._get_slot(event.get("output_index", 0), "toolCall")
            if slot is not None and slot.partial_json is not None:
                delta = event.get("delta") or ""
                slot.partial_json += delta
                slot.block.arguments = parse_streaming_json(slot.partial_json)
                yield ToolCallDeltaEvent(
                    content_index=slot.content_index, delta=delta, partial=self.output
                )

        elif event_type == "response.function_call_arguments.done":
            slot = self._get_slot(event.get("output_index", 0), "toolCall")
            if slot is not None and slot.partial_json is not None:
                previous = slot.partial_json
                slot.partial_json = event.get("arguments") or ""
                slot.block.arguments = parse_streaming_json(slot.partial_json)
                # 只有在最终值是已累积值的延伸时才补发差量，否则会发出错乱的片段
                if slot.partial_json.startswith(previous):
                    delta = slot.partial_json[len(previous) :]
                    if delta:
                        yield ToolCallDeltaEvent(
                            content_index=slot.content_index, delta=delta, partial=self.output
                        )

        elif event_type == "response.custom_tool_call_input.delta":
            slot = self._get_slot(event.get("output_index", 0), "toolCall")
            if slot is not None and slot.custom_property is not None:
                next_input = slot.current_custom_input() + (event.get("delta") or "")
                delta = self._append_custom_input(slot, next_input, False)
                if delta is not None:
                    yield ToolCallDeltaEvent(
                        content_index=slot.content_index, delta=delta, partial=self.output
                    )

        elif event_type == "response.custom_tool_call_input.done":
            slot = self._get_slot(event.get("output_index", 0), "toolCall")
            if slot is not None and slot.custom_property is not None:
                delta = self._append_custom_input(slot, event.get("input") or "", True)
                if delta is not None:
                    yield ToolCallDeltaEvent(
                        content_index=slot.content_index, delta=delta, partial=self.output
                    )

        elif event_type == "response.output_item.done":
            yield from self._handle_item_done(event)

        elif event_type in ("response.completed", "response.incomplete"):
            self._finalize(event.get("response") or {})

        elif event_type == "error":
            code = event.get("code")
            message = event.get("message")
            raise ProviderHTTPError(f"Error Code {code}: {message}")

        elif event_type == "response.failed":
            self.saw_terminal_event = True
            response = event.get("response") or {}
            self.output.raw_stop_reason = response.get("status")
            raise ProviderHTTPError(_describe_failure(response))

    def _handle_item_done(self, event: Mapping[str, Any]) -> Iterator[AssistantMessageEvent]:
        item = event.get("item") or {}
        output_index = event.get("output_index", 0)
        self._apply_message_phase(item)

        slot = self._slots.get(output_index)
        if slot is None:
            slot, start_event = self._create_slot(output_index, item)
            if start_event is not None:
                yield start_event
        if slot is None:
            return

        item_type = item.get("type")

        if item_type == "reasoning" and slot.kind == "thinking":
            summary_text = "\n\n".join(s.get("text", "") for s in (item.get("summary") or []))
            content_text = "\n\n".join(c.get("text", "") for c in (item.get("content") or []))
            slot.block.thinking = summary_text or content_text or slot.block.thinking
            slot.block.thinking_signature = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            if item.get("id"):
                self._reasoning_blocks_by_id[item["id"]] = slot.block
            yield ThinkingEndEvent(
                content_index=slot.content_index, content=slot.block.thinking, partial=self.output
            )
            self._slots.pop(output_index, None)

        elif item_type == "message" and slot.kind == "text":
            parts = item.get("content") or []
            slot.block.text = "".join(
                (p.get("text") or "") if p.get("type") == "output_text" else (p.get("refusal") or "")
                for p in parts
            )
            slot.block.text_signature = encode_text_signature_v1(item.get("id") or "", item.get("phase"))
            yield TextEndEvent(
                content_index=slot.content_index, content=slot.block.text, partial=self.output
            )
            self._slots.pop(output_index, None)

        elif item_type == "function_call" and slot.kind == "toolCall" and slot.partial_json is not None:
            slot.block.arguments = parse_streaming_json(item.get("arguments") or slot.partial_json or "{}")
            if item.get("namespace") is not None:
                slot.block.namespace = item["namespace"]
            slot.partial_json = None
            yield ToolCallEndEvent(
                content_index=slot.content_index, tool_call=slot.block, partial=self.output
            )
            self._slots.pop(output_index, None)

        elif item_type == "custom_tool_call" and slot.kind == "toolCall" and slot.custom_property is not None:
            final_input = item.get("input")
            if final_input is None:
                final_input = slot.current_custom_input()
            delta = self._append_custom_input(slot, final_input, True)
            if delta is not None:
                yield ToolCallDeltaEvent(
                    content_index=slot.content_index, delta=delta, partial=self.output
                )
            if item.get("namespace") is not None:
                slot.block.namespace = item["namespace"]
            slot.custom_property = None
            slot.json_buffer = None
            yield ToolCallEndEvent(
                content_index=slot.content_index, tool_call=slot.block, partial=self.output
            )
            self._slots.pop(output_index, None)

    def _backfill_reasoning_signatures(self, response_output: list[Mapping[str, Any]]) -> None:
        """从终态响应回填加密推理内容。

        Azure 可能在 output_item.done 里省略 encrypted_content，只在
        response.completed 的 output 里给。不回填的话 store:false 的多轮
        回放会丢掉加密推理，导致后续请求被拒。
        """
        for item in response_output:
            if item.get("type") != "reasoning" or not item.get("encrypted_content"):
                continue
            block = self._reasoning_blocks_by_id.get(item.get("id") or "")
            if block is None or not block.thinking_signature:
                continue
            try:
                stored = json.loads(block.thinking_signature)
            except ValueError:
                continue
            if stored.get("encrypted_content"):
                continue
            stored["encrypted_content"] = item["encrypted_content"]
            block.thinking_signature = json.dumps(stored, ensure_ascii=False, separators=(",", ":"))

    def _finalize(self, response: Mapping[str, Any]) -> None:
        self.saw_terminal_event = True
        self._backfill_reasoning_signatures(list(response.get("output") or []))

        if response.get("id"):
            self.output.response_id = response["id"]

        usage = response.get("usage")
        if usage:
            details = usage.get("input_tokens_details") or {}
            cached = details.get("cached_tokens") or 0
            cache_write = details.get("cache_write_tokens") or 0
            output_details = usage.get("output_tokens_details") or {}
            self.output.usage = Usage(
                # OpenAI 把缓存读写都算进 input_tokens，两者都要减掉
                input=max(0, (usage.get("input_tokens") or 0) - cached - cache_write),
                output=usage.get("output_tokens") or 0,
                cache_read=cached,
                cache_write=cache_write,
                reasoning=output_details.get("reasoning_tokens") or 0,
                total_tokens=usage.get("total_tokens") or 0,
            )

        calculate_cost(self.model, self.output.usage)
        if self.apply_service_tier_pricing is not None:
            if self.resolve_service_tier is not None:
                tier = self.resolve_service_tier(response.get("service_tier"), self.service_tier)
            else:
                tier = response.get("service_tier") or self.service_tier
            self.apply_service_tier_pricing(self.output.usage, tier)

        status = response.get("status")
        incomplete_details = response.get("incomplete_details") or {}
        incomplete_reason = incomplete_details.get("reason")
        if not isinstance(incomplete_reason, str):
            incomplete_reason = None

        self.output.raw_stop_reason = f"{status}.{incomplete_reason}" if incomplete_reason else status
        stop_reason, error_message = map_stop_reason(status, incomplete_reason)
        self.output.stop_reason = stop_reason
        self.output.error_message = error_message

        if any(b.type == "toolCall" for b in self.output.content) and self.output.stop_reason == "stop":
            self.output.stop_reason = "toolUse"


def _describe_failure(response: Mapping[str, Any]) -> str:
    error = response.get("error")
    if error:
        return f"{error.get('code') or 'unknown'}: {error.get('message') or 'no message'}"
    details = response.get("incomplete_details") or {}
    if details.get("reason"):
        return f"incomplete: {details['reason']}"
    return "Unknown error (no error details in response)"
