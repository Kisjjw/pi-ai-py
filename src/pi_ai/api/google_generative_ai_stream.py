from __future__ import annotations

import itertools
import json
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from pi_ai.api.google_shared import is_thinking_part, map_stop_reason, retain_thought_signature
from pi_ai.cost import calculate_cost
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Model,
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

_tool_call_counter = itertools.count(1)


@dataclass
class GoogleStreamParser:
    """把 Gemini 的 chunk 解析成统一事件。

    与其它适配器的差别：这里的块是**按类型连续合并**的——同类 part 累加进
    当前块，类型切换时先给前一个块发 end 再开新块。没有 index 或槽位概念。
    """

    model: Model
    _current_block: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.output = AssistantMessage(
            api=self.model.api,
            provider=self.model.provider,
            model=self.model.id,
            stop_reason="pending",
        )

    def _block_index(self) -> int:
        return len(self.output.content) - 1

    def _close_current_block(self) -> Iterator[AssistantMessageEvent]:
        block = self._current_block
        if block is None:
            return
        index = self.output.content.index(block)
        if block.type == "text":
            yield TextEndEvent(content_index=index, content=block.text, partial=self.output)
        else:
            yield ThinkingEndEvent(content_index=index, content=block.thinking, partial=self.output)
        self._current_block = None

    def feed(self, chunk: Mapping[str, Any]) -> Iterator[AssistantMessageEvent]:
        if not isinstance(chunk, Mapping):
            return

        if not self.output.response_id and chunk.get("responseId"):
            self.output.response_id = chunk["responseId"]

        candidates = chunk.get("candidates") or []
        candidate = candidates[0] if candidates else None

        if candidate:
            parts = (candidate.get("content") or {}).get("parts") or []
            for part in parts:
                yield from self._handle_part(part)

            if candidate.get("finishReason"):
                self.output.raw_stop_reason = candidate["finishReason"]
                self.output.stop_reason = map_stop_reason(candidate["finishReason"])
                if any(b.type == "toolCall" for b in self.output.content):
                    self.output.stop_reason = "toolUse"

        usage_metadata = chunk.get("usageMetadata")
        if usage_metadata:
            self._apply_usage(usage_metadata)

    def _handle_part(self, part: Mapping[str, Any]) -> Iterator[AssistantMessageEvent]:
        if part.get("text") is not None:
            yield from self._handle_text_part(part)
        if part.get("functionCall"):
            yield from self._handle_function_call(part)

    def _handle_text_part(self, part: Mapping[str, Any]) -> Iterator[AssistantMessageEvent]:
        is_thinking = is_thinking_part(part)
        current = self._current_block

        needs_new_block = (
            current is None
            or (is_thinking and current.type != "thinking")
            or (not is_thinking and current.type != "text")
        )
        if needs_new_block:
            yield from self._close_current_block()
            if is_thinking:
                block: Any = ThinkingContent(thinking="")
                self.output.content.append(block)
                self._current_block = block
                yield ThinkingStartEvent(content_index=self._block_index(), partial=self.output)
            else:
                block = TextContent(text="")
                self.output.content.append(block)
                self._current_block = block
                yield TextStartEvent(content_index=self._block_index(), partial=self.output)

        block = self._current_block
        text = part["text"]
        signature = part.get("thoughtSignature")

        if block.type == "thinking":
            block.thinking += text
            block.thinking_signature = retain_thought_signature(block.thinking_signature, signature)
            yield ThinkingDeltaEvent(
                content_index=self._block_index(), delta=text, partial=self.output
            )
        else:
            block.text += text
            block.text_signature = retain_thought_signature(block.text_signature, signature)
            yield TextDeltaEvent(content_index=self._block_index(), delta=text, partial=self.output)

    def _handle_function_call(self, part: Mapping[str, Any]) -> Iterator[AssistantMessageEvent]:
        yield from self._close_current_block()

        function_call = part["functionCall"]
        name = function_call.get("name") or ""
        provided_id = function_call.get("id")
        # id 缺失或与已有重复时自己生成一个，否则回放时会撞车
        needs_new_id = not provided_id or any(
            b.type == "toolCall" and b.id == provided_id for b in self.output.content
        )
        call_id = (
            f"{name}_{int(time.time() * 1000)}_{next(_tool_call_counter)}" if needs_new_id else provided_id
        )

        tool_call = ToolCall(
            id=call_id,
            name=name,
            arguments=function_call.get("args") or {},
            thought_signature=part.get("thoughtSignature"),
        )
        self.output.content.append(tool_call)
        index = self._block_index()

        # Google 不做函数调用的流式增量，一次性三连发完
        yield ToolCallStartEvent(content_index=index, partial=self.output)
        yield ToolCallDeltaEvent(
            content_index=index,
            delta=json.dumps(tool_call.arguments, ensure_ascii=False, separators=(",", ":")),
            partial=self.output,
        )
        yield ToolCallEndEvent(content_index=index, tool_call=tool_call, partial=self.output)

    def _apply_usage(self, metadata: Mapping[str, Any]) -> None:
        prompt = metadata.get("promptTokenCount") or 0
        cached = metadata.get("cachedContentTokenCount") or 0
        candidates = metadata.get("candidatesTokenCount") or 0
        thoughts = metadata.get("thoughtsTokenCount") or 0

        self.output.usage = Usage(
            input=prompt - cached,
            # 与其它厂商不同：Google 的 candidatesTokenCount 不含思考 token，要自己加上
            output=candidates + thoughts,
            cache_read=cached,
            cache_write=0,
            reasoning=thoughts,
            total_tokens=metadata.get("totalTokenCount") or 0,
        )
        calculate_cost(self.model, self.output.usage)

    def finish(self) -> Iterator[AssistantMessageEvent]:
        yield from self._close_current_block()
