from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Wire(BaseModel):
    """所有线上类型的基类。

    ConfigDict 必须内联写在这里，不能引用模块级变量：pydantic 的 mypy 插件
    是静态分析的，解析不了变量引用，就会不知道 populate_by_name=True，
    进而要求构造对象时使用线上 alias 名。
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")


def now_ms() -> int:
    return int(time.time() * 1000)


# --- 内容块 ---


class TextContent(_Wire):
    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = Field(default=None, alias="textSignature")


class ThinkingContent(_Wire):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = Field(default=None, alias="thinkingSignature")
    redacted: bool | None = None


class ImageContent(_Wire):
    type: Literal["image"] = "image"
    data: str
    mime_type: str = Field(alias="mimeType")


class ToolCall(_Wire):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    thought_signature: str | None = Field(default=None, alias="thoughtSignature")
    namespace: str | None = None


AssistantContent = Annotated[
    TextContent | ThinkingContent | ToolCall,
    Field(discriminator="type"),
]
UserContent = Annotated[TextContent | ImageContent, Field(discriminator="type")]


# --- 用量与费用 ---


class CostBreakdown(_Wire):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")
    total: float = 0.0


class Usage(_Wire):
    input: int = 0
    output: int = 0
    cache_read: int = Field(default=0, alias="cacheRead")
    cache_write: int = Field(default=0, alias="cacheWrite")
    cache_write_1h: int | None = Field(default=None, alias="cacheWrite1h")
    reasoning: int | None = None
    total_tokens: int = Field(default=0, alias="totalTokens")
    cost: CostBreakdown = Field(default_factory=CostBreakdown)


StopReason = Literal["pending", "stop", "length", "toolUse", "error", "aborted", "deferred"]


# --- 消息 ---


class UserMessage(_Wire):
    role: Literal["user"] = "user"
    content: str | list[UserContent]
    timestamp: int = Field(default_factory=now_ms)


class AssistantMessage(_Wire):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    api: str
    provider: str
    model: str
    response_model: str | None = Field(default=None, alias="responseModel")
    response_id: str | None = Field(default=None, alias="responseId")
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = Field(default="pending", alias="stopReason")
    error_message: str | None = Field(default=None, alias="errorMessage")
    raw_stop_reason: str | None = Field(default=None, alias="rawStopReason")
    end_turn: bool | None = Field(default=None, alias="endTurn")
    timestamp: int = Field(default_factory=now_ms)


class ToolResultMessage(_Wire):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    content: list[UserContent] = Field(default_factory=list)
    details: Any = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = Field(default=None, alias="addedToolNames")
    is_error: bool = Field(default=False, alias="isError")
    timestamp: int = Field(default_factory=now_ms)


Message = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]


def user_text(text: str) -> UserMessage:
    return UserMessage(content=text)


# --- 工具与上下文 ---


class JsonSchemaSampling(_Wire):
    type: Literal["json_schema"] = "json_schema"
    strict: Literal["prefer", "require"]


class GrammarSampling(_Wire):
    type: Literal["grammar"] = "grammar"
    variants: dict[str, str] = Field(default_factory=dict)


ConstrainedSampling = Annotated[
    JsonSchemaSampling | GrammarSampling,
    Field(discriminator="type"),
]


class Tool(_Wire):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    constrained_sampling: ConstrainedSampling | Literal[False] | None = Field(
        default=None, alias="constrainedSampling"
    )

    @classmethod
    def from_pydantic(cls, model_cls: type[BaseModel], name: str, description: str = "") -> Tool:
        return cls(name=name, description=description, parameters=model_cls.model_json_schema())


class Context(_Wire):
    system_prompt: str | None = Field(default=None, alias="systemPrompt")
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] | None = None


# --- 模型与兼容开关 ---


class OpenAICompletionsCompat(_Wire):
    supports_store: bool | None = Field(default=None, alias="supportsStore")
    supports_developer_role: bool | None = Field(default=None, alias="supportsDeveloperRole")
    supports_reasoning_effort: bool | None = Field(default=None, alias="supportsReasoningEffort")
    supports_usage_in_streaming: bool | None = Field(default=None, alias="supportsUsageInStreaming")
    supports_finish_reason: bool | None = Field(default=None, alias="supportsFinishReason")
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] | None = Field(
        default=None, alias="maxTokensField"
    )
    requires_tool_result_name: bool | None = Field(default=None, alias="requiresToolResultName")
    requires_assistant_after_tool_result: bool | None = Field(
        default=None, alias="requiresAssistantAfterToolResult"
    )
    requires_thinking_as_text: bool | None = Field(default=None, alias="requiresThinkingAsText")
    requires_reasoning_content_on_assistant_messages: bool | None = Field(
        default=None, alias="requiresReasoningContentOnAssistantMessages"
    )
    thinking_format: str | None = Field(default=None, alias="thinkingFormat")
    chat_template_kwargs: dict[str, Any] | None = Field(default=None, alias="chatTemplateKwargs")
    chat_template_args: dict[str, Any] | None = Field(default=None, alias="chatTemplateArgs")
    open_router_routing: dict[str, Any] | None = Field(default=None, alias="openRouterRouting")
    vercel_gateway_routing: dict[str, Any] | None = Field(default=None, alias="vercelGatewayRouting")
    zai_tool_stream: bool | None = Field(default=None, alias="zaiToolStream")
    supports_thinking_token_budget: bool | None = Field(default=None, alias="supportsThinkingTokenBudget")
    supports_openai_grammar_tools: bool | None = Field(default=None, alias="supportsOpenAIGrammarTools")
    supports_strict_mode: bool | None = Field(default=None, alias="supportsStrictMode")
    cache_control_format: Literal["anthropic"] | None = Field(default=None, alias="cacheControlFormat")
    send_session_affinity_headers: bool | None = Field(default=None, alias="sendSessionAffinityHeaders")
    deferred_tools_mode: Literal["kimi"] | None = Field(default=None, alias="deferredToolsMode")
    session_affinity_format: str | None = Field(default=None, alias="sessionAffinityFormat")
    supports_long_cache_retention: bool | None = Field(default=None, alias="supportsLongCacheRetention")


class OpenAIResponsesCompat(_Wire):
    supports_developer_role: bool | None = Field(default=None, alias="supportsDeveloperRole")
    # 生成的模型目录会给部分 openai-responses 模型带上这个字段，尽管 TS 的
    # OpenAIResponsesCompat 接口并未声明它。这里如实建模，但适配器不据此改变行为。
    supports_reasoning_effort: bool | None = Field(default=None, alias="supportsReasoningEffort")
    session_affinity_format: str | None = Field(default=None, alias="sessionAffinityFormat")
    supports_long_cache_retention: bool | None = Field(default=None, alias="supportsLongCacheRetention")
    supports_strict_mode: bool | None = Field(default=None, alias="supportsStrictMode")
    supports_openai_grammar_tools: bool | None = Field(default=None, alias="supportsOpenAIGrammarTools")
    supports_additional_tools: bool | None = Field(default=None, alias="supportsAdditionalTools")
    supports_tool_search: bool | None = Field(default=None, alias="supportsToolSearch")
    supports_explicit_prompt_cache_mode: bool | None = Field(
        default=None, alias="supportsExplicitPromptCacheMode"
    )


class AnthropicMessagesCompat(_Wire):
    supports_eager_tool_input_streaming: bool | None = Field(
        default=None, alias="supportsEagerToolInputStreaming"
    )
    supports_long_cache_retention: bool | None = Field(default=None, alias="supportsLongCacheRetention")
    send_session_affinity_headers: bool | None = Field(default=None, alias="sendSessionAffinityHeaders")
    supports_cache_control_on_tools: bool | None = Field(default=None, alias="supportsCacheControlOnTools")
    supports_temperature: bool | None = Field(default=None, alias="supportsTemperature")
    force_adaptive_thinking: bool | None = Field(default=None, alias="forceAdaptiveThinking")
    allow_empty_signature: bool | None = Field(default=None, alias="allowEmptySignature")
    supports_strict_tools: bool | None = Field(default=None, alias="supportsStrictTools")
    supports_tool_references: bool | None = Field(default=None, alias="supportsToolReferences")


class BedrockCompat(_Wire):
    """Amazon Bedrock 的兼容开关。

    Bedrock 适配器本身属于 L3，但生成的模型目录已经包含带 compat 的 Bedrock 模型，
    因此类型必须先支持解析，否则加载 amazon-bedrock 目录会直接失败。
    """

    supports_strict_mode: bool | None = Field(default=None, alias="supportsStrictMode")


Compat = OpenAICompletionsCompat | OpenAIResponsesCompat | AnthropicMessagesCompat | BedrockCompat

_COMPAT_BY_API: dict[str, type[BaseModel]] = {
    "openai-completions": OpenAICompletionsCompat,
    "openai-responses": OpenAIResponsesCompat,
    "azure-openai-responses": OpenAIResponsesCompat,
    "openai-codex-responses": OpenAIResponsesCompat,
    "anthropic-messages": AnthropicMessagesCompat,
    "bedrock-converse-stream": BedrockCompat,
}


def _default_input() -> list[Literal["text", "image"]]:
    return ["text"]


class ModelCostRates(_Wire):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")


class ModelCostTier(ModelCostRates):
    input_tokens_above: int = Field(alias="inputTokensAbove")


class ModelCost(ModelCostRates):
    tiers: list[ModelCostTier] | None = None


class Model(_Wire):
    id: str
    name: str
    api: str
    provider: str
    base_url: str = Field(alias="baseUrl")
    reasoning: bool = False
    thinking_level_map: dict[str, str | None] | None = Field(default=None, alias="thinkingLevelMap")
    input: list[Literal["text", "image"]] = Field(default_factory=_default_input)
    cost: ModelCost = Field(default_factory=ModelCost)
    context_window: int = Field(default=0, alias="contextWindow")
    max_tokens: int = Field(default=0, alias="maxTokens")
    sampling_params: dict[str, Any] | None = Field(default=None, alias="samplingParams")
    headers: dict[str, str] | None = None
    compat: Any = None

    @model_validator(mode="after")
    def _parse_compat(self) -> Model:
        if self.compat is None or isinstance(self.compat, BaseModel):
            return self
        compat_cls = _COMPAT_BY_API.get(self.api)
        if compat_cls is None:
            raise ValueError(f"api {self.api!r} does not support compat settings")
        object.__setattr__(self, "compat", compat_cls.model_validate(self.compat))
        return self


# --- 事件 ---


class StartEvent(_Wire):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextStartEvent(_Wire):
    type: Literal["text_start"] = "text_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class TextDeltaEvent(_Wire):
    type: Literal["text_delta"] = "text_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class TextEndEvent(_Wire):
    type: Literal["text_end"] = "text_end"
    content_index: int = Field(alias="contentIndex")
    content: str
    partial: AssistantMessage


class ThinkingStartEvent(_Wire):
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class ThinkingDeltaEvent(_Wire):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(_Wire):
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int = Field(alias="contentIndex")
    content: str
    partial: AssistantMessage


class ToolCallStartEvent(_Wire):
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class ToolCallDeltaEvent(_Wire):
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(_Wire):
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int = Field(alias="contentIndex")
    tool_call: ToolCall = Field(alias="toolCall")
    partial: AssistantMessage


class DoneEvent(_Wire):
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse", "deferred"]
    message: AssistantMessage


class ErrorEvent(_Wire):
    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent = Annotated[
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent,
    Field(discriminator="type"),
]
