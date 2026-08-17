from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pi_ai.api.constrained_sampling import resolve_json_schema_strict_sampling
from pi_ai.api.transform_messages import transform_messages
from pi_ai.types import (
    AssistantMessage,
    Context,
    Model,
    StopReason,
    Tool,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.utils.sanitize import sanitize_surrogates

#: Gemini 3 的思考级别，对应 Google 的 ThinkingLevel 枚举取值。
GOOGLE_THINKING_LEVELS = ("THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH")

#: 思考签名必须是合法 base64（Google 的 TYPE_BYTES）。
_BASE64_SIGNATURE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")

_GEMINI_MAJOR = re.compile(r"^gemini(?:-live)?-(\d+)")
_UNSAFE_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_TOOL_CALL_ID_MAX_LENGTH = 64

TOOL_RESULT_IMAGE_PLACEHOLDER = "(see attached image)"
TOOL_RESULT_IMAGE_PREFIX = "Tool result image:"

#: JSON Schema 的元声明，转成 OpenAPI 风格时要剥掉。
_JSON_SCHEMA_META_DECLARATIONS = frozenset(
    {"$schema", "$id", "$anchor", "$dynamicAnchor", "$vocabulary", "$comment", "$defs", "definitions"}
)


def is_thinking_part(part: Mapping[str, Any]) -> bool:
    """判断一个 Gemini part 是否属于思考内容。

    `thought: true` 才是思考标记。`thoughtSignature` 可以出现在任意 part 上
    （文本、函数调用都行），**不代表该 part 是思考内容**——这是最容易搞错的一处。

    见 https://ai.google.dev/gemini-api/docs/thought-signatures
    """
    return part.get("thought") is True


def retain_thought_signature(existing: str | None, incoming: str | None) -> str | None:
    """流式过程中保住签名。

    某些后端只在某个 part 的第一个增量里带 thoughtSignature，后续增量省略。
    这个函数只防止同一个块内的签名被 None 覆盖，**不会跨 part 搬运签名**。
    """
    if isinstance(incoming, str) and incoming:
        return incoming
    return existing


def is_valid_thought_signature(signature: str | None) -> bool:
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return bool(_BASE64_SIGNATURE.match(signature))


def resolve_thought_signature(is_same_provider_and_model: bool, signature: str | None) -> str | None:
    """只保留同 provider 同 model 且 base64 合法的签名。"""
    return signature if is_same_provider_and_model and is_valid_thought_signature(signature) else None


def get_gemini_major_version(model_id: str) -> int | None:
    match = _GEMINI_MAJOR.match(model_id.lower())
    return int(match.group(1)) if match else None


def requires_tool_call_id(model_id: str) -> bool:
    """哪些经 Google API 提供的模型要求函数调用/响应里显式带 id。"""
    major = get_gemini_major_version(model_id)
    return (
        model_id.startswith("claude-")
        or model_id.startswith("gpt-oss-")
        or (major is not None and major >= 3)
    )


def supports_multimodal_function_response(model_id: str) -> bool:
    """Gemini 3+ 支持把图片嵌进 functionResponse.parts。"""
    major = get_gemini_major_version(model_id)
    if major is not None:
        return major >= 3
    return True


def supports_google_strict_tool_sampling(model_id: str) -> bool:
    """Gemini 3+ 才会在受验证的工具调用模式下强制必填参数。"""
    major = get_gemini_major_version(model_id)
    return major is not None and major >= 3


def convert_messages(model: Model, context: Context) -> list[dict[str, Any]]:
    """把会话历史转成 Gemini 的 Content 列表。"""
    contents: list[dict[str, Any]] = []

    def normalize_tool_call_id(call_id: str, _model: Model, _source: AssistantMessage) -> str:
        if not requires_tool_call_id(model.id):
            return call_id
        return _UNSAFE_ID_CHARS.sub("_", call_id)[:_TOOL_CALL_ID_MAX_LENGTH]

    transformed = transform_messages(context.messages, model, normalize_tool_call_id)

    for msg in transformed:
        if isinstance(msg, UserMessage):
            _append_user(contents, msg)
        elif isinstance(msg, AssistantMessage):
            _append_assistant(contents, msg, model)
        elif isinstance(msg, ToolResultMessage):
            _append_tool_result(contents, msg, model)

    return contents


def _append_user(contents: list[dict[str, Any]], msg: UserMessage) -> None:
    if isinstance(msg.content, str):
        contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(msg.content)}]})
        return

    parts: list[dict[str, Any]] = []
    for item in msg.content:
        if item.type == "text":
            parts.append({"text": sanitize_surrogates(item.text)})
        else:
            parts.append({"inlineData": {"mimeType": item.mime_type, "data": item.data}})
    if not parts:
        return
    contents.append({"role": "user", "parts": parts})


def _append_assistant(contents: list[dict[str, Any]], msg: AssistantMessage, model: Model) -> None:
    parts: list[dict[str, Any]] = []
    # 只有同 provider 且同 model 时才保留思考块与签名
    is_same = msg.provider == model.provider and msg.model == model.id
    needs_id = requires_tool_call_id(model.id)

    for block in msg.content:
        if block.type == "text":
            signature = resolve_thought_signature(is_same, block.text_signature)
            # 空文本块只有在不带签名时才丢。Gemini 会把签名挂在可见文本为空的
            # part 上并要求原样回传，丢了会断掉推理链——模型会间歇性地以
            # "只有思考、没有内容也没有工具调用"的 STOP 结束回合。
            if (not block.text or not block.text.strip()) and not signature:
                continue
            part: dict[str, Any] = {"text": sanitize_surrogates(block.text)}
            if signature:
                part["thoughtSignature"] = signature
            parts.append(part)
            continue

        if block.type == "thinking":
            if is_same:
                signature = resolve_thought_signature(is_same, block.thinking_signature)
                # 与文本块同一条规则：空思考块只在没有签名时才丢
                if (not block.thinking or not block.thinking.strip()) and not signature:
                    continue
                part = {"thought": True, "text": sanitize_surrogates(block.thinking)}
                if signature:
                    part["thoughtSignature"] = signature
                parts.append(part)
            else:
                # 跨 provider/model：签名不可用，空块照丢，其余降级成普通文本
                if not block.thinking or not block.thinking.strip():
                    continue
                parts.append({"text": sanitize_surrogates(block.thinking)})
            continue

        if block.type == "toolCall":
            signature = resolve_thought_signature(is_same, block.thought_signature)
            function_call: dict[str, Any] = {"name": block.name, "args": block.arguments or {}}
            if needs_id:
                function_call["id"] = block.id
            part = {"functionCall": function_call}
            if signature:
                part["thoughtSignature"] = signature
            parts.append(part)

    if not parts:
        return
    contents.append({"role": "model", "parts": parts})


def _append_tool_result(contents: list[dict[str, Any]], msg: ToolResultMessage, model: Model) -> None:
    text_result = "\n".join(b.text for b in msg.content if b.type == "text")
    images = [b for b in msg.content if b.type == "image"] if "image" in model.input else []

    has_text = bool(text_result)
    has_images = bool(images)
    multimodal = supports_multimodal_function_response(model.id)

    # 成功用 output 键，失败用 error 键
    response_value = sanitize_surrogates(text_result) if has_text else (
        TOOL_RESULT_IMAGE_PLACEHOLDER if has_images else ""
    )
    image_parts = [{"inlineData": {"mimeType": b.mime_type, "data": b.data}} for b in images]

    function_response: dict[str, Any] = {
        "name": msg.tool_name,
        "response": {"error": response_value} if msg.is_error else {"output": response_value},
    }
    if has_images and multimodal:
        function_response["parts"] = image_parts
    if requires_tool_call_id(model.id):
        function_response["id"] = msg.tool_call_id

    part = {"functionResponse": function_response}

    # Cloud Code Assist 要求所有函数响应挤在同一个 user turn 里
    last = contents[-1] if contents else None
    if last is not None and last.get("role") == "user" and any("functionResponse" in p for p in last["parts"]):
        last["parts"].append(part)
    else:
        contents.append({"role": "user", "parts": [part]})

    # Gemini 3 之前的版本不支持多模态函数响应，图片另发一条 user 消息
    if has_images and not multimodal:
        contents.append({"role": "user", "parts": [{"text": TOOL_RESULT_IMAGE_PREFIX}, *image_parts]})


def _sanitize_for_openapi(schema: Any) -> Any:
    """剥掉 JSON Schema 的元声明，OpenAPI 3.03 不认它们。"""
    if not isinstance(schema, dict):
        return schema
    return {
        key: _sanitize_for_openapi(value)
        for key, value in schema.items()
        if key not in _JSON_SCHEMA_META_DECLARATIONS
    }


def convert_tools(tools: Sequence[Tool], use_parameters: bool = False) -> list[dict[str, Any]] | None:
    """把工具定义转成 Gemini 的 functionDeclarations。

    默认用 parametersJsonSchema（支持完整 JSON Schema，含 anyOf/oneOf/const）。
    use_parameters 为真时改用旧的 parameters 字段（OpenAPI 3.03 Schema），
    Cloud Code Assist 上的 Claude 模型需要这种形式，因为那边会把 parameters
    翻译成 Anthropic 的 input_schema。
    """
    if not tools:
        return None

    declarations: list[dict[str, Any]] = []
    for tool in tools:
        declaration: dict[str, Any] = {"name": tool.name, "description": tool.description}
        if use_parameters:
            declaration["parameters"] = _sanitize_for_openapi(tool.parameters)
        else:
            declaration["parametersJsonSchema"] = tool.parameters
        declarations.append(declaration)

    return [{"functionDeclarations": declarations}]


def map_tool_choice(choice: str) -> str:
    if choice == "none":
        return "NONE"
    if choice == "any":
        return "ANY"
    return "AUTO"


def resolve_google_function_calling_mode(
    tools: Sequence[Tool],
    tool_choice: str | None,
    supports_strict_mode: bool,
) -> str | None:
    use_strict = any(resolve_json_schema_strict_sampling(t, supports_strict_mode) is True for t in tools)
    if tool_choice in ("none", "any"):
        return map_tool_choice(tool_choice)
    if use_strict:
        return "VALIDATED"
    return map_tool_choice(tool_choice) if tool_choice else None


#: 除 STOP / MAX_TOKENS 外的 finishReason 全部映射成 error。
_STOP_REASONS: dict[str, StopReason] = {"STOP": "stop", "MAX_TOKENS": "length"}


def map_stop_reason(reason: str) -> StopReason:
    return _STOP_REASONS.get(reason, "error")
