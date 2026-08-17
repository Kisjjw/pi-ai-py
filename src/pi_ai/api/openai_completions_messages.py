from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pi_ai.api.constrained_sampling import (
    get_grammar_tool_input,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from pi_ai.api.openai_completions_compat import ResolvedCompat
from pi_ai.api.transform_messages import transform_messages
from pi_ai.types import (
    AssistantMessage,
    Context,
    Message,
    Model,
    Tool,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.utils.hash import short_hash
from pi_ai.utils.sanitize import sanitize_surrogates

_TOOL_CALL_ID_MAX_LENGTH = 40
_UNSAFE_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

SYNTHETIC_ASSISTANT_TEXT = "I have processed the tool results."
TOOL_RESULT_IMAGE_PLACEHOLDER = "(see attached image)"
TOOL_RESULT_EMPTY_PLACEHOLDER = "(no tool output)"
TOOL_RESULT_IMAGE_PREFIX = "Attached image(s) from tool result:"


def normalize_tool_call_id(model: Model, call_id: str) -> str:
    """把工具调用 id 归一成 Chat Completions 能接受的形状。

    Responses API 会产出 `{call_id}|{item_id}` 形式的复合 id，其中 item_id
    可能长达 400 字符且带 `+` `/` `=`。同一回合的多个调用可能共享 call_id，
    只有 item_id 不同，所以不能只取前半段——那会让 id 撞车。
    """
    if "|" in call_id:
        separator_index = call_id.index("|")
        call_part = _UNSAFE_ID_CHARS.sub("_", call_id[:separator_index])
        item_part = _UNSAFE_ID_CHARS.sub("_", call_id[separator_index + 1 :])
        combined = f"{call_part}_{item_part}" if item_part else call_part
        if len(combined) <= _TOOL_CALL_ID_MAX_LENGTH:
            return combined
        digest = short_hash(call_id)[:8]
        prefix = call_part[: max(1, _TOOL_CALL_ID_MAX_LENGTH - len(digest) - 1)]
        return f"{prefix}_{digest}"

    if model.provider == "openai":
        return call_id[:_TOOL_CALL_ID_MAX_LENGTH] if len(call_id) > _TOOL_CALL_ID_MAX_LENGTH else call_id
    return call_id


def has_tool_history(messages: Sequence[Message]) -> bool:
    """会话里是否出现过工具调用或工具结果。

    经代理转发的 Anthropic 要求消息里有 tool_calls 时请求必须带 tools 字段，
    哪怕是空数组。
    """
    for msg in messages:
        if isinstance(msg, ToolResultMessage):
            return True
        if isinstance(msg, AssistantMessage) and any(b.type == "toolCall" for b in msg.content):
            return True
    return False


def get_deferred_tool_names(messages: Sequence[Message]) -> set[str]:
    names: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolResultMessage):
            names.update(msg.added_tool_names or [])
    return names


def get_tools_by_name(tools: Sequence[Tool] | None, names: Sequence[str] | set[str]) -> list[Tool]:
    if not tools:
        return []
    by_name = {tool.name: tool for tool in tools}
    return [by_name[name] for name in names if name in by_name]


def convert_tools(tools: Sequence[Tool], compat: ResolvedCompat) -> list[dict[str, Any]]:
    """把统一的工具定义转成 OpenAI 的 tools 数组。"""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, compat.supports_openai_grammar_tools)
        if grammar is not None:
            converted.append(
                {
                    "type": "custom",
                    "custom": {
                        "name": tool.name,
                        "description": tool.description,
                        "format": {
                            "type": "grammar",
                            "grammar": {"syntax": grammar.format, "definition": grammar.definition},
                        },
                    },
                }
            )
            continue

        strict = resolve_json_schema_strict_sampling(tool, compat.supports_strict_mode is not False)
        function: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        # 不支持 strict 的厂商会因为未知字段直接拒绝请求，所以整个字段都要省掉。
        if compat.supports_strict_mode is not False:
            function["strict"] = strict if strict is not None else False
        converted.append({"type": "function", "function": function})
    return converted


def get_compat_cache_control(compat: ResolvedCompat, cache_retention: str) -> dict[str, Any] | None:
    if compat.cache_control_format != "anthropic" or cache_retention == "none":
        return None
    if cache_retention == "long" and compat.supports_long_cache_retention:
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _add_cache_control_to_text_content(message: dict[str, Any], cache_control: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
        return True

    if not isinstance(content, list):
        return False

    for part in reversed(content):
        if isinstance(part, dict) and part.get("type") == "text":
            part["cache_control"] = cache_control
            return True
    return False


def apply_anthropic_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cache_control: dict[str, Any],
) -> None:
    """在 OpenAI 兼容端点上打 Anthropic 风格的缓存标记。

    落点三处：系统提示、工具列表最后一个、以及从后往前第一个能加上的对话消息。
    """
    for message in messages:
        if message.get("role") in ("system", "developer"):
            _add_cache_control_to_text_content(message, cache_control)
            break

    if tools:
        tools[-1]["cache_control"] = cache_control

    for message in reversed(messages):
        if message.get("role") in ("user", "assistant", "tool"):
            if _add_cache_control_to_text_content(message, cache_control):
                return


def convert_messages(
    model: Model,
    context: Context,
    compat: ResolvedCompat,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """把统一的会话上下文转成 OpenAI Chat Completions 的 messages 数组。

    移植自 TS 的 openai-completions.ts:1046-1335。
    """
    grammar_tool_input_properties = grammar_tool_input_properties or {}
    params: list[dict[str, Any]] = []

    transformed = transform_messages(
        context.messages, model, lambda call_id, m, _source: normalize_tool_call_id(m, call_id)
    )

    if context.system_prompt:
        use_developer_role = model.reasoning and compat.supports_developer_role
        params.append(
            {
                "role": "developer" if use_developer_role else "system",
                "content": sanitize_surrogates(context.system_prompt),
            }
        )

    last_role: str | None = None
    index = 0
    while index < len(transformed):
        msg = transformed[index]

        # 部分厂商不允许工具结果后直接跟 user 消息，插一条合成 assistant 过渡。
        if compat.requires_assistant_after_tool_result and last_role == "toolResult" and msg.role == "user":
            params.append({"role": "assistant", "content": SYNTHETIC_ASSISTANT_TEXT})

        if isinstance(msg, UserMessage):
            _append_user_message(params, msg)
        elif isinstance(msg, AssistantMessage):
            _append_assistant_message(params, msg, model, compat, grammar_tool_input_properties)
        elif isinstance(msg, ToolResultMessage):
            index, last_role = _append_tool_results(
                params, transformed, index, model, context, compat
            )
            index += 1
            continue

        last_role = msg.role
        index += 1

    return params


def _append_user_message(params: list[dict[str, Any]], msg: UserMessage) -> None:
    if isinstance(msg.content, str):
        params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
        return

    content: list[dict[str, Any]] = []
    for item in msg.content:
        if item.type == "text":
            content.append({"type": "text", "text": sanitize_surrogates(item.text)})
        else:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"}}
            )
    if not content:
        return
    params.append({"role": "user", "content": content})


def _append_assistant_message(
    params: list[dict[str, Any]],
    msg: AssistantMessage,
    model: Model,
    compat: ResolvedCompat,
    grammar_tool_input_properties: Mapping[str, str],
) -> None:
    assistant: dict[str, Any] = {
        "role": "assistant",
        # 部分厂商不接受 null content，用空字符串代替。
        "content": "" if compat.requires_assistant_after_tool_result else None,
    }

    text_parts = [
        {"type": "text", "text": sanitize_surrogates(b.text)}
        for b in msg.content
        if b.type == "text" and b.text.strip()
    ]
    assistant_text = "".join(part["text"] for part in text_parts)

    thinking_blocks = [b for b in msg.content if b.type == "thinking" and b.thinking.strip()]

    if thinking_blocks:
        if compat.requires_thinking_as_text:
            # 转成纯文本，不加标签，避免模型模仿标签结构
            thinking_text = "\n\n".join(sanitize_surrogates(b.thinking) for b in thinking_blocks)
            assistant["content"] = [{"type": "text", "text": thinking_text}, *text_parts]
        else:
            # assistant 内容必须是纯字符串。发成 [{type:"text"}] 数组是非标准的，
            # 会让某些模型（例如 NVIDIA NIM 上的 DeepSeek V3.2）把结构原样吐回来。
            if assistant_text:
                assistant["content"] = assistant_text

            signature = thinking_blocks[0].thinking_signature
            if model.provider == "opencode-go" and signature == "reasoning":
                signature = "reasoning_content"
            if signature:
                assistant[signature] = "\n".join(b.thinking for b in thinking_blocks)
    elif assistant_text:
        assistant["content"] = assistant_text

    tool_calls = [b for b in msg.content if b.type == "toolCall"]
    if tool_calls:
        serialized: list[dict[str, Any]] = []
        for call in tool_calls:
            custom_property = grammar_tool_input_properties.get(call.name)
            if custom_property is not None:
                serialized.append(
                    {
                        "id": call.id,
                        "type": "custom",
                        "custom": {
                            "name": call.name,
                            "input": sanitize_surrogates(
                                get_grammar_tool_input(call.name, call.arguments, custom_property)
                            ),
                        },
                    }
                )
            else:
                serialized.append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")),
                        },
                    }
                )
        assistant["tool_calls"] = serialized

        reasoning_details = []
        for call in tool_calls:
            if not call.thought_signature:
                continue
            try:
                reasoning_details.append(json.loads(call.thought_signature))
            except ValueError:
                continue
        if reasoning_details:
            assistant["reasoning_details"] = reasoning_details

    if (
        compat.requires_reasoning_content_on_assistant_messages
        and model.reasoning
        and "reasoning_content" not in assistant
    ):
        assistant["reasoning_content"] = ""

    # 既无内容又无工具调用的 assistant 消息要整条跳过：不少厂商要求
    # "content 与 tool_calls 至少有一个"。中断的回合会产生这种消息。
    content = assistant.get("content")
    has_content = content is not None and len(content) > 0
    if not has_content and "tool_calls" not in assistant:
        return

    params.append(assistant)


def _append_tool_results(
    params: list[dict[str, Any]],
    transformed: Sequence[Message],
    start_index: int,
    model: Model,
    context: Context,
    compat: ResolvedCompat,
) -> tuple[int, str]:
    """处理一段连续的工具结果，返回最后一条的下标与新的 last_role。"""
    image_blocks: list[dict[str, Any]] = []
    deferred_tool_names: set[str] = set()

    index = start_index
    while index < len(transformed):
        tool_msg = transformed[index]
        if not isinstance(tool_msg, ToolResultMessage):
            break

        text_result = "\n".join(b.text for b in tool_msg.content if b.type == "text")
        has_images = any(b.type == "image" for b in tool_msg.content)
        if text_result:
            tool_result_text = text_result
        elif has_images:
            tool_result_text = TOOL_RESULT_IMAGE_PLACEHOLDER
        else:
            tool_result_text = TOOL_RESULT_EMPTY_PLACEHOLDER

        tool_result: dict[str, Any] = {
            "role": "tool",
            "content": sanitize_surrogates(tool_result_text),
            "tool_call_id": tool_msg.tool_call_id,
        }
        if compat.requires_tool_result_name and tool_msg.tool_name:
            tool_result["name"] = tool_msg.tool_name
        params.append(tool_result)

        if compat.deferred_tools_mode == "kimi":
            deferred_tool_names.update(tool_msg.added_tool_names or [])

        if has_images and "image" in model.input:
            for block in tool_msg.content:
                if block.type == "image":
                    image_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                        }
                    )

        index += 1

    last_index = index - 1

    if image_blocks:
        if compat.requires_assistant_after_tool_result:
            params.append({"role": "assistant", "content": SYNTHETIC_ASSISTANT_TEXT})
        params.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": TOOL_RESULT_IMAGE_PREFIX}, *image_blocks],
            }
        )
        last_role = "user"
    else:
        last_role = "toolResult"

    if deferred_tool_names:
        deferred_tools = get_tools_by_name(context.tools, deferred_tool_names)
        if deferred_tools:
            # Kimi 接受一条带 tools 的 system 消息，且不带标准的 content 字段。
            params.append({"role": "system", "tools": convert_tools(deferred_tools, compat)})

    return last_index, last_role
