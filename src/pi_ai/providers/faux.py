"""按脚本产出事件序列的假 provider。

不发网络请求、不需要 API key，供依赖本库的上层项目（比如 agent 循环）写测试。
对应 TS 的 src/providers/faux.ts。

用法：

    streams = FauxStreams([
        FauxResponse(text="hi"),
        FauxResponse(tool_calls=[FauxToolCall(id="c1", name="read", arguments={"path": "a"})]),
    ])
    models = create_models()
    models.set_provider(faux_provider(streams))
    stream = models.stream_simple(faux_model(), context)

streams.requests 记录了每次调用收到的 model / context / options，可用来断言选项传递。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pi_ai.auth.types import ApiKeyCredential, AuthContext, AuthResult, ModelAuth
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.provider import Provider, create_provider
from pi_ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
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

FAUX_API = "faux"
FAUX_PROVIDER_ID = "faux"

FauxStopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


@dataclass
class FauxToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class FauxResponse:
    """一轮响应的脚本。

    stop_reason 留空时自动推断：有工具调用为 "toolUse"，否则 "stop"。
    """

    text: str | None = None
    thinking: str | None = None
    tool_calls: Sequence[FauxToolCall] = ()
    stop_reason: FauxStopReason | None = None
    error_message: str | None = None
    usage: Usage | None = None
    #: >0 时把文本、思考、工具参数按此长度切成多个增量，用于测试增量事件。
    chunk_size: int = 0
    #: 每个增量之间的等待秒数。给取消与并发测试留出插入点。
    delay: float = 0.0


@dataclass
class FauxRequest:
    """记录一次调用，供测试断言选项传递是否正确。"""

    method: Literal["stream", "stream_simple"]
    model: Model
    context: Context
    options: dict[str, Any]


class FauxStreams:
    """ProviderStreams 实现。按 responses 顺序消费。

    脚本用尽后产出一条 stop_reason="error" 的响应而不是重复最后一条：重复会让「循环
    多跑了几轮」这类 bug 变成无限循环，而报错能立刻定位。
    """

    def __init__(self, responses: Sequence[FauxResponse] | None = None) -> None:
        self.responses: list[FauxResponse] = list(responses or [])
        self.requests: list[FauxRequest] = []
        self._index = 0

    def queue(self, *responses: FauxResponse) -> None:
        self.responses.extend(responses)

    @property
    def consumed(self) -> int:
        return self._index

    def stream(self, model: Model, context: Context, options: Mapping[str, Any]) -> AssistantMessageEventStream:
        return self._start("stream", model, context, options)

    def stream_simple(self, model: Model, context: Context, options: Mapping[str, Any]) -> AssistantMessageEventStream:
        return self._start("stream_simple", model, context, options)

    def _start(
        self,
        method: Literal["stream", "stream_simple"],
        model: Model,
        context: Context,
        options: Mapping[str, Any],
    ) -> AssistantMessageEventStream:
        self.requests.append(FauxRequest(method=method, model=model, context=context, options=dict(options)))
        response = self._next_response()

        stream = AssistantMessageEventStream()
        task = asyncio.ensure_future(_produce(stream, model, response))
        # bind_producer 让 cancel() 取消生产者任务，任务内部捕获 CancelledError 后
        # 产出 aborted 事件，与真实适配器的取消行为一致。
        stream.bind_producer(task)
        return stream

    def _next_response(self) -> FauxResponse:
        if self._index < len(self.responses):
            response = self.responses[self._index]
            self._index += 1
            return response
        return FauxResponse(
            stop_reason="error",
            error_message=f"faux provider ran out of scripted responses (consumed {self._index})",
        )


@dataclass
class FauxApiKeyAuth:
    """总是解析成功的认证，不读环境变量、不需要凭据。"""

    name: str = "Faux API key"

    async def resolve(self, *, ctx: AuthContext, credential: ApiKeyCredential | None) -> AuthResult | None:
        return AuthResult(auth=ModelAuth(api_key="faux-key"), source="faux")


def faux_model(
    model_id: str = "faux-model",
    *,
    name: str | None = None,
    reasoning: bool = True,
    context_window: int = 128_000,
    max_tokens: int = 8_192,
) -> Model:
    return Model(
        id=model_id,
        name=name or model_id,
        api=FAUX_API,
        provider=FAUX_PROVIDER_ID,
        base_url="",
        reasoning=reasoning,
        input=["text", "image"],
        context_window=context_window,
        max_tokens=max_tokens,
    )


def faux_provider(streams: FauxStreams | None = None, models: Sequence[Model] | None = None) -> Provider:
    return create_provider(
        id=FAUX_PROVIDER_ID,
        name="Faux",
        base_url="",
        auth={"api_key": FauxApiKeyAuth()},
        models=list(models or [faux_model()]),
        api=streams or FauxStreams(),
    )


# --- 事件产出 ---


async def _produce(stream: AssistantMessageEventStream, model: Model, response: FauxResponse) -> None:
    partial = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="pending",
    )
    try:
        stream.push(StartEvent(partial=partial))

        if response.thinking is not None:
            await _emit_thinking(stream, partial, response)
        if response.text is not None:
            await _emit_text(stream, partial, response)
        for spec in response.tool_calls:
            await _emit_tool_call(stream, partial, response, spec)

        partial.usage = response.usage or Usage()
        reason: FauxStopReason = response.stop_reason or ("toolUse" if response.tool_calls else "stop")

        if reason == "error" or reason == "aborted":
            partial.stop_reason = reason
            partial.error_message = response.error_message or f"faux {reason}"
            stream.push(ErrorEvent(reason=reason, error=partial))
        else:
            partial.stop_reason = reason
            stream.push(DoneEvent(reason=reason, message=partial))
    except asyncio.CancelledError:
        # 与真实适配器一致：取消时产出带已收增量的 aborted 消息，再让取消继续传播。
        partial.stop_reason = "aborted"
        partial.error_message = "Request aborted by user"
        stream.push(ErrorEvent(reason="aborted", error=partial))
        raise


async def _emit_text(
    stream: AssistantMessageEventStream,
    partial: AssistantMessage,
    response: FauxResponse,
) -> None:
    index = len(partial.content)
    block = TextContent(text="")
    partial.content.append(block)
    stream.push(TextStartEvent(content_index=index, partial=partial))

    for chunk in _chunks(response.text or "", response.chunk_size):
        if response.delay:
            await asyncio.sleep(response.delay)
        block.text += chunk
        stream.push(TextDeltaEvent(content_index=index, delta=chunk, partial=partial))

    stream.push(TextEndEvent(content_index=index, content=block.text, partial=partial))


async def _emit_thinking(
    stream: AssistantMessageEventStream,
    partial: AssistantMessage,
    response: FauxResponse,
) -> None:
    index = len(partial.content)
    block = ThinkingContent(thinking="")
    partial.content.append(block)
    stream.push(ThinkingStartEvent(content_index=index, partial=partial))

    for chunk in _chunks(response.thinking or "", response.chunk_size):
        if response.delay:
            await asyncio.sleep(response.delay)
        block.thinking += chunk
        stream.push(ThinkingDeltaEvent(content_index=index, delta=chunk, partial=partial))

    stream.push(ThinkingEndEvent(content_index=index, content=block.thinking, partial=partial))


async def _emit_tool_call(
    stream: AssistantMessageEventStream,
    partial: AssistantMessage,
    response: FauxResponse,
    spec: FauxToolCall,
) -> None:
    index = len(partial.content)
    block = ToolCall(id=spec.id, name=spec.name, arguments={})
    partial.content.append(block)
    stream.push(ToolCallStartEvent(content_index=index, partial=partial))

    payload = json.dumps(spec.arguments)
    accumulated = ""
    for chunk in _chunks(payload, response.chunk_size):
        if response.delay:
            await asyncio.sleep(response.delay)
        accumulated += chunk
        # 用真实适配器同一个容错解析器，让 arguments 的中间态与线上一致。
        block.arguments = parse_streaming_json(accumulated) or {}
        stream.push(ToolCallDeltaEvent(content_index=index, delta=chunk, partial=partial))

    block.arguments = dict(spec.arguments)
    stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=partial))


def _chunks(text: str, size: int) -> list[str]:
    if not text:
        return []
    if size <= 0:
        return [text]
    return [text[index : index + size] for index in range(0, len(text), size)]
