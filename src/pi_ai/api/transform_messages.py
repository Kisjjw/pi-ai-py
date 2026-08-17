from __future__ import annotations

from collections.abc import Callable, Sequence

from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    now_ms,
)

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"

NormalizeToolCallId = Callable[[str, Model, AssistantMessage], str]


def _replace_images_with_placeholder(
    content: Sequence[TextContent | ImageContent],
    placeholder: str,
) -> list[TextContent]:
    """把图片换成占位文本，连续多张只留一个占位。"""
    result: list[TextContent] = []
    previous_was_placeholder = False

    for block in content:
        if block.type == "image":
            if not previous_was_placeholder:
                result.append(TextContent(text=placeholder))
            previous_was_placeholder = True
            continue

        result.append(block)
        previous_was_placeholder = block.text == placeholder

    return result


def _downgrade_unsupported_images(messages: Sequence[Message], model: Model) -> list[Message]:
    if "image" in model.input:
        return list(messages)

    downgraded: list[Message] = []
    for msg in messages:
        if isinstance(msg, UserMessage) and not isinstance(msg.content, str):
            downgraded.append(
                msg.model_copy(
                    update={
                        "content": _replace_images_with_placeholder(
                            msg.content, NON_VISION_USER_IMAGE_PLACEHOLDER
                        )
                    }
                )
            )
        elif isinstance(msg, ToolResultMessage):
            downgraded.append(
                msg.model_copy(
                    update={
                        "content": _replace_images_with_placeholder(
                            msg.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER
                        )
                    }
                )
            )
        else:
            downgraded.append(msg)
    return downgraded


def transform_messages(
    messages: Sequence[Message],
    model: Model,
    normalize_tool_call_id: NormalizeToolCallId | None = None,
) -> list[Message]:
    """把会话历史转换成可安全回放给目标模型的形态。

    移植自 TS 的 src/api/transform-messages.ts。做四件事：
    非视觉模型的图片降级、跨模型的思考块与签名处理、工具调用 id 归一化、
    以及给孤立的工具调用补上合成结果。
    """
    tool_call_id_map: dict[str, str] = {}

    normalized: list[Message] = [
        msg.model_copy(update={"content": []}) if getattr(msg, "content", ...) is None else msg for msg in messages
    ]
    image_aware = _downgrade_unsupported_images(normalized, model)

    transformed: list[Message] = []
    for msg in image_aware:
        if isinstance(msg, UserMessage):
            transformed.append(msg)
            continue

        if isinstance(msg, ToolResultMessage):
            mapped = tool_call_id_map.get(msg.tool_call_id)
            if mapped is not None and mapped != msg.tool_call_id:
                transformed.append(msg.model_copy(update={"tool_call_id": mapped}))
            else:
                transformed.append(msg)
            continue

        is_same_model = msg.provider == model.provider and msg.api == model.api and msg.model == model.id

        content: list[TextContent | object] = []
        for block in msg.content:
            if block.type == "thinking":
                # redacted thinking 是加密的不透明内容，只对同一模型有效
                if block.redacted:
                    if is_same_model:
                        content.append(block)
                    continue
                # 同模型下带签名的思考块必须保留，哪怕文本为空（OpenAI 的加密推理）
                if is_same_model and block.thinking_signature:
                    content.append(block)
                    continue
                if not block.thinking or not block.thinking.strip():
                    continue
                if is_same_model:
                    content.append(block)
                else:
                    content.append(TextContent(text=block.thinking))
                continue

            if block.type == "text":
                content.append(block if is_same_model else TextContent(text=block.text))
                continue

            if block.type == "toolCall":
                tool_call: ToolCall = block
                if not is_same_model and tool_call.thought_signature:
                    tool_call = tool_call.model_copy(update={"thought_signature": None})

                if not is_same_model and normalize_tool_call_id is not None:
                    normalized_id = normalize_tool_call_id(tool_call.id, model, msg)
                    if normalized_id != tool_call.id:
                        tool_call_id_map[tool_call.id] = normalized_id
                        tool_call = tool_call.model_copy(update={"id": normalized_id})

                content.append(tool_call)
                continue

            content.append(block)

        transformed.append(msg.model_copy(update={"content": content}))

    return _insert_synthetic_tool_results(transformed)


def _insert_synthetic_tool_results(messages: Sequence[Message]) -> list[Message]:
    """给没有结果的工具调用补上合成结果。

    某些 API 要求每个 tool_call 都必须有对应的 tool result，否则报错。
    补入时机有三个：下一条 assistant 之前、下一条 user 之前、以及遍历结束。
    """
    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_result_ids: set[str] = set()

    def flush() -> None:
        nonlocal pending_tool_calls, existing_result_ids
        if not pending_tool_calls:
            return
        for tool_call in pending_tool_calls:
            if tool_call.id not in existing_result_ids:
                result.append(
                    ToolResultMessage(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=[TextContent(text="No result provided")],
                        is_error=True,
                        timestamp=now_ms(),
                    )
                )
        pending_tool_calls = []
        existing_result_ids = set()

    for msg in messages:
        if isinstance(msg, AssistantMessage):
            flush()
            # error / aborted 是未完成的回合，回放它们会引发 API 错误
            if msg.stop_reason in ("error", "aborted"):
                continue
            tool_calls = [b for b in msg.content if b.type == "toolCall"]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_result_ids = set()
            result.append(msg)
        elif isinstance(msg, ToolResultMessage):
            existing_result_ids.add(msg.tool_call_id)
            result.append(msg)
        elif isinstance(msg, UserMessage):
            flush()
            result.append(msg)
        else:
            result.append(msg)

    flush()
    return result
