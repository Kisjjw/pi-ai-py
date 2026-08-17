from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from pi_ai.api.constrained_sampling import resolve_json_schema_strict_sampling
from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    Tool,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.utils.sanitize import sanitize_surrogates

#: 隐身模式：完全模仿 Claude Code 的工具命名。
CLAUDE_CODE_VERSION = "2.1.75"

#: Claude Code 2.x 的工具名（规范大小写）。
#: 来源 https://cchistory.mariozechner.at/data/prompts-2.1.11.md
CLAUDE_CODE_TOOLS = (
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
)

_CC_TOOL_LOOKUP = {name.lower(): name for name in CLAUDE_CODE_TOOLS}

_UNSAFE_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_TOOL_CALL_ID_MAX_LENGTH = 64

IMAGE_ONLY_PLACEHOLDER = "(see attached image)"


def to_claude_code_name(name: str) -> str:
    """工具名大小写不敏感地映射到 Claude Code 的规范写法。"""
    return _CC_TOOL_LOOKUP.get(name.lower(), name)


def from_claude_code_name(name: str, tools: Sequence[Tool] | None = None) -> str:
    """把模型回传的规范名映射回调用方定义的工具名。"""
    if tools:
        lowered = name.lower()
        for tool in tools:
            if tool.name.lower() == lowered:
                return tool.name
    return name


def normalize_tool_call_id(call_id: str) -> str:
    """归一成 Anthropic 要求的字符集与长度。"""
    return _UNSAFE_ID_CHARS.sub("_", call_id)[:_TOOL_CALL_ID_MAX_LENGTH]


def convert_content_blocks(
    content: Sequence[TextContent | ImageContent],
) -> str | list[dict[str, Any]]:
    """把内容块转成 Anthropic 格式。

    全是文本时直接返回拼接的字符串（Anthropic 接受这种简写）；
    含图片时返回块数组，且若一张文本都没有，前面补一个占位文本块。
    """
    has_images = any(block.type == "image" for block in content)
    if not has_images:
        return sanitize_surrogates("\n".join(b.text for b in content if b.type == "text"))

    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": block.mime_type, "data": block.data},
                }
            )

    if not any(b["type"] == "text" for b in blocks):
        blocks.insert(0, {"type": "text", "text": IMAGE_ONLY_PLACEHOLDER})
    return blocks


def convert_tool_result(
    msg: ToolResultMessage,
    is_oauth: bool,
    deferred_tool_names: set[str],
    loaded_tool_names: set[str],
    normalize_tool_name: Callable[[str], str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """转换一条工具结果，返回 (tool_result 块, 需要挪出去的兄弟内容)。

    Anthropic 不接受 tool_reference 与普通结果内容混在同一个 tool_result 里，
    所以带引用时把普通内容挪到兄弟位置。
    """
    references: list[dict[str, Any]] = []
    for name in msg.added_tool_names or []:
        normalized = normalize_tool_name(name)
        if normalized not in deferred_tool_names or normalized in loaded_tool_names:
            continue
        loaded_tool_names.add(normalized)
        references.append(
            {"type": "tool_reference", "tool_name": to_claude_code_name(name) if is_oauth else name}
        )

    converted = convert_content_blocks(msg.content)

    tool_result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": references if references else converted,
        "is_error": msg.is_error,
    }

    if not references:
        sibling: list[dict[str, Any]] = []
    elif isinstance(converted, str):
        sibling = [{"type": "text", "text": converted}]
    else:
        sibling = converted

    return tool_result, sibling


def convert_messages(
    transformed_messages: Sequence[Message],
    is_oauth: bool,
    cache_control: dict[str, Any] | None = None,
    allow_empty_signature: bool = False,
    deferred_tool_names: set[str] | None = None,
    normalize_tool_name: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """把会话历史转成 Anthropic 的 messages 数组。"""
    deferred_tool_names = deferred_tool_names or set()
    normalize = normalize_tool_name or (lambda name: name)
    params: list[dict[str, Any]] = []
    loaded_tool_names: set[str] = set()

    index = 0
    while index < len(transformed_messages):
        msg = transformed_messages[index]

        if isinstance(msg, UserMessage):
            _append_user(params, msg)
        elif isinstance(msg, AssistantMessage):
            _append_assistant(params, msg, is_oauth, allow_empty_signature)
        elif isinstance(msg, ToolResultMessage):
            index = _append_tool_results(
                params, transformed_messages, index, is_oauth, deferred_tool_names, loaded_tool_names, normalize
            )

        index += 1

    if cache_control and params:
        _mark_last_user_message(params[-1], cache_control)

    return params


def _append_user(params: list[dict[str, Any]], msg: UserMessage) -> None:
    if isinstance(msg.content, str):
        if msg.content.strip():
            params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
        return

    blocks: list[dict[str, Any]] = []
    for item in msg.content:
        if item.type == "text":
            text = sanitize_surrogates(item.text)
            if text.strip():
                blocks.append({"type": "text", "text": text})
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": item.mime_type, "data": item.data},
                }
            )
    if not blocks:
        return
    params.append({"role": "user", "content": blocks})


def _append_assistant(
    params: list[dict[str, Any]],
    msg: AssistantMessage,
    is_oauth: bool,
    allow_empty_signature: bool,
) -> None:
    blocks: list[dict[str, Any]] = []

    for block in msg.content:
        if block.type == "text":
            if not block.text.strip():
                continue
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
            continue

        if block.type == "thinking":
            # redacted 思考是不透明的加密载荷，原样发回去
            if block.redacted:
                blocks.append({"type": "redacted_thinking", "data": block.thinking_signature})
                continue

            signature = block.thinking_signature
            has_signature = bool(signature and signature.strip())
            if not block.thinking.strip() and not has_signature:
                continue

            if not has_signature:
                # 签名缺失（例如中断的流）时 Anthropic 会拒绝该块。部分兼容厂商
                # 会发出并接受空签名，允许它们保留原块。
                if allow_empty_signature:
                    blocks.append(
                        {
                            "type": "thinking",
                            "thinking": sanitize_surrogates(block.thinking),
                            "signature": "",
                        }
                    )
                else:
                    blocks.append({"type": "text", "text": sanitize_surrogates(block.thinking)})
            else:
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": sanitize_surrogates(block.thinking),
                        "signature": signature,
                    }
                )
            continue

        if block.type == "toolCall":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": to_claude_code_name(block.name) if is_oauth else block.name,
                    "input": block.arguments or {},
                }
            )

    if not blocks:
        return
    params.append({"role": "assistant", "content": blocks})


def _append_tool_results(
    params: list[dict[str, Any]],
    transformed: Sequence[Message],
    start_index: int,
    is_oauth: bool,
    deferred_tool_names: set[str],
    loaded_tool_names: set[str],
    normalize: Callable[[str], str],
) -> int:
    """把连续的工具结果合并成一条 user 消息，返回最后一条的下标。"""
    tool_results: list[dict[str, Any]] = []
    sibling_content: list[dict[str, Any]] = []

    index = start_index
    while index < len(transformed):
        msg = transformed[index]
        if not isinstance(msg, ToolResultMessage):
            break
        tool_result, sibling = convert_tool_result(
            msg, is_oauth, deferred_tool_names, loaded_tool_names, normalize
        )
        tool_results.append(tool_result)
        sibling_content.extend(sibling)
        index += 1

    # 被挪出来的、带引用的结果内容必须排在所有 tool_result 之后。
    params.append({"role": "user", "content": [*tool_results, *sibling_content]})
    return index - 1


def _mark_last_user_message(message: dict[str, Any], cache_control: dict[str, Any]) -> None:
    """给最后一条 user 消息打缓存标记，用于缓存整段对话历史。"""
    if message.get("role") != "user":
        return

    content = message.get("content")
    if isinstance(content, list):
        if not content:
            return
        last_block = content[-1]
        if isinstance(last_block, dict) and last_block.get("type") in ("text", "image", "tool_result"):
            last_block["cache_control"] = cache_control
    elif isinstance(content, str):
        message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]


def convert_tools(
    tools: Sequence[Tool],
    is_oauth: bool,
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: dict[str, Any] | None = None,
    defer_loading: bool = False,
) -> list[dict[str, Any]]:
    """把统一的工具定义转成 Anthropic 的 tools 数组。"""
    converted: list[dict[str, Any]] = []
    total = len(tools)

    for index, tool in enumerate(tools):
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_tools)
        schema = tool.parameters or {}
        legacy_input_schema = {
            "type": "object",
            "properties": schema.get("properties") or {},
            "required": schema.get("required") or [],
        }
        # strict 模式下发完整 schema，但 legacy 三件套在后，会覆盖同名键。
        input_schema = {**schema, **legacy_input_schema} if strict is True else legacy_input_schema

        entry: dict[str, Any] = {
            "name": to_claude_code_name(tool.name) if is_oauth else tool.name,
            "description": tool.description,
        }
        if supports_eager_tool_input_streaming:
            entry["eager_input_streaming"] = True
        if strict is True:
            entry["strict"] = True
        entry["input_schema"] = input_schema
        if defer_loading:
            entry["defer_loading"] = True
        if cache_control and index == total - 1:
            entry["cache_control"] = cache_control

        converted.append(entry)

    return converted
