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
from pi_ai.api.transform_messages import transform_messages
from pi_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    TextContent,
    Tool,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.utils.hash import short_hash
from pi_ai.utils.sanitize import sanitize_surrogates

_UNSAFE_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_TRAILING_UNDERSCORES = re.compile(r"_+$")
_ID_MAX_LENGTH = 64

TOOL_RESULT_IMAGE_PLACEHOLDER = "(see attached image)"
TOOL_RESULT_EMPTY_PLACEHOLDER = "(no tool output)"


# --- 文本签名 ---


def encode_text_signature_v1(item_id: str, phase: str | None = None) -> str:
    payload: dict[str, Any] = {"v": 1, "id": item_id}
    if phase:
        payload["phase"] = phase
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_text_signature(signature: str | None) -> dict[str, Any] | None:
    """解析文本块签名。兼容旧版的裸 id 字符串。"""
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
        except ValueError:
            return {"id": signature}
        if isinstance(parsed, dict) and parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
            if parsed.get("phase") in ("commentary", "final_answer"):
                return {"id": parsed["id"], "phase": parsed["phase"]}
            return {"id": parsed["id"]}
    return {"id": signature}


# --- id 归一化 ---


def normalize_id_part(part: str) -> str:
    sanitized = _UNSAFE_ID_CHARS.sub("_", part)[:_ID_MAX_LENGTH]
    return _TRAILING_UNDERSCORES.sub("", sanitized)


def build_foreign_responses_item_id(item_id: str) -> str:
    return f"fc_{short_hash(item_id)}"[:_ID_MAX_LENGTH]


def make_tool_call_id_normalizer(
    model: Model,
    allowed_tool_call_providers: frozenset[str] | set[str],
) -> Any:
    """构造 transform_messages 用的 id 归一化回调。

    Responses 的工具调用 id 是 `callId|itemId` 复合形式。只有白名单 provider
    才保留这个结构，其余一律压平成单段 id。
    """

    def normalize(call_id: str, _target: Model, source: AssistantMessage) -> str:
        if model.provider not in allowed_tool_call_providers:
            return normalize_id_part(call_id)
        if "|" not in call_id:
            return normalize_id_part(call_id)

        call_part, _, item_part = call_id.partition("|")
        normalized_call = normalize_id_part(call_part)
        is_foreign = source.provider != model.provider or source.api != model.api
        normalized_item = (
            build_foreign_responses_item_id(item_part) if is_foreign else normalize_id_part(item_part)
        )
        # Responses API 要求 item id 以 fc_ 开头
        if not normalized_item.startswith("fc_"):
            normalized_item = normalize_id_part(f"fc_{normalized_item}")
        return f"{normalized_call}|{normalized_item}"

    return normalize


# --- 工具结果 ---


def convert_tool_result_output(
    model: Model,
    content: Sequence[TextContent | ImageContent],
) -> str | list[dict[str, Any]]:
    text_result = "\n".join(b.text for b in content if b.type == "text")
    images = [b for b in content if b.type == "image"]
    has_text = bool(text_result)

    if not images or "image" not in model.input:
        if has_text:
            return sanitize_surrogates(text_result)
        return TOOL_RESULT_IMAGE_PLACEHOLDER if images else TOOL_RESULT_EMPTY_PLACEHOLDER

    output: list[dict[str, Any]] = []
    if has_text:
        output.append({"type": "input_text", "text": sanitize_surrogates(text_result)})
    for image in images:
        output.append(
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{image.mime_type};base64,{image.data}",
            }
        )
    return output


# --- 工具转换 ---


def convert_responses_tools(
    tools: Sequence[Tool],
    *,
    strict: bool | None = None,
    supports_strict_mode: bool = True,
    supports_openai_grammar_tools: bool = False,
    defer_loading: bool = False,
) -> list[dict[str, Any]]:
    """把统一的工具定义转成 Responses 的 tools 数组。

    注意语法工具的 format 结构与 Chat Completions 不同：这里是平铺的
    {type, syntax, definition}，那边是嵌套的 {type, grammar:{...}}。
    """
    default_strict = False if strict is None else strict
    converted: list[dict[str, Any]] = []

    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar is not None:
            entry: dict[str, Any] = {
                "type": "custom",
                "name": tool.name,
                "description": tool.description,
                "format": {
                    "type": "grammar",
                    "syntax": grammar.format,
                    "definition": grammar.definition,
                },
            }
            if defer_loading:
                entry["defer_loading"] = True
            converted.append(entry)
            continue

        constrained_strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        function_tool: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if defer_loading:
            function_tool["defer_loading"] = True
        if supports_strict_mode:
            function_tool["strict"] = constrained_strict if constrained_strict is not None else default_strict
        converted.append(function_tool)

    return converted


# --- 消息转换 ---


def convert_responses_messages(
    model: Model,
    context: Context,
    allowed_tool_call_providers: frozenset[str] | set[str],
    *,
    include_system_prompt: bool = True,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
    deferred_tools: Mapping[str, Tool] | None = None,
    deferred_tools_mode: str | None = None,
    tool_options: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """把会话历史转成 Responses 的 input item 列表。

    与 Chat Completions 的最大差别：输出的是扁平的 item 序列，一条 assistant
    消息可能展开成多个 item（推理 item、消息 item、工具调用 item）。
    """
    grammar_tool_input_properties = grammar_tool_input_properties or {}
    tool_options = dict(tool_options or {})
    messages: list[dict[str, Any]] = []
    loaded_tool_names: set[str] = set()

    transformed = transform_messages(
        context.messages, model, make_tool_call_id_normalizer(model, allowed_tool_call_providers)
    )

    if include_system_prompt and context.system_prompt:
        supports_developer_role = getattr(model.compat, "supports_developer_role", None)
        role = "developer" if model.reasoning and supports_developer_role is not False else "system"
        messages.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    for msg_index, msg in enumerate(transformed):
        if isinstance(msg, UserMessage):
            _append_user(messages, msg)
        elif isinstance(msg, AssistantMessage):
            _append_assistant(
                messages, msg, model, msg_index, grammar_tool_input_properties, deferred_tools
            )
        elif isinstance(msg, ToolResultMessage):
            _append_tool_result(
                messages,
                msg,
                model,
                grammar_tool_input_properties,
                deferred_tools,
                deferred_tools_mode,
                tool_options,
                loaded_tool_names,
            )

    return messages


def _append_user(messages: list[dict[str, Any]], msg: UserMessage) -> None:
    if isinstance(msg.content, str):
        messages.append(
            {"role": "user", "content": [{"type": "input_text", "text": sanitize_surrogates(msg.content)}]}
        )
        return

    content: list[dict[str, Any]] = []
    for item in msg.content:
        if item.type == "text":
            content.append({"type": "input_text", "text": sanitize_surrogates(item.text)})
        else:
            content.append(
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": f"data:{item.mime_type};base64,{item.data}",
                }
            )
    if not content:
        return
    messages.append({"role": "user", "content": content})


def _append_assistant(
    messages: list[dict[str, Any]],
    msg: AssistantMessage,
    model: Model,
    msg_index: int,
    grammar_tool_input_properties: Mapping[str, str],
    deferred_tools: Mapping[str, Tool] | None,
) -> None:
    output: list[dict[str, Any]] = []
    is_same_provider_and_api = msg.provider == model.provider and msg.api == model.api
    is_same_model = is_same_provider_and_api and msg.model == model.id
    is_different_model = is_same_provider_and_api and msg.model != model.id
    text_block_index = 0

    for block in msg.content:
        if block.type == "thinking":
            # 签名本身就是序列化的 reasoning item，整个塞回去
            if block.thinking_signature:
                try:
                    output.append(json.loads(block.thinking_signature))
                except ValueError:
                    continue
            continue

        if block.type == "text":
            parsed = parse_text_signature(block.text_signature)
            fallback_id = (
                f"msg_pi_{msg_index}" if text_block_index == 0 else f"msg_pi_{msg_index}_{text_block_index}"
            )
            text_block_index += 1

            msg_id = (parsed or {}).get("id") or fallback_id
            if len(msg_id) > _ID_MAX_LENGTH:
                msg_id = f"msg_{short_hash(msg_id)}"

            output.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": sanitize_surrogates(block.text), "annotations": []}
                    ],
                    "status": "completed",
                    "id": msg_id,
                    "phase": (parsed or {}).get("phase"),
                }
            )
            continue

        if block.type == "toolCall":
            call_part, _, item_part = block.id.partition("|")
            custom_property = grammar_tool_input_properties.get(block.name)
            item_id: str | None = item_part or None

            # 两种情况要丢掉 item id：
            # 1) 同 provider 但换了模型 —— OpenAI 会校验 fc_xxx 与 rs_xxx 的配对，
            #    省略 id 就绕开该校验（跨 provider 本来就走这条路）
            # 2) 把自定义工具调用回放成 function_call 时，id 必须是 fc_* 开头，
            #    ctc_* 这类自定义工具 id 不合法
            if (is_different_model and item_id and item_id.startswith("fc_")) or (
                custom_property is None and not (item_id or "").startswith("fc_")
            ):
                item_id = None

            can_replay_namespace = is_same_model or (
                deferred_tools is not None and block.name in deferred_tools
            )
            namespace_field = (
                {"namespace": block.namespace}
                if can_replay_namespace and block.namespace is not None
                else {}
            )

            if custom_property is not None:
                output.append(
                    {
                        "type": "custom_tool_call",
                        "id": item_id,
                        "call_id": call_part,
                        "name": block.name,
                        "input": sanitize_surrogates(
                            get_grammar_tool_input(block.name, block.arguments, custom_property)
                        ),
                        **namespace_field,
                    }
                )
            else:
                output.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_part,
                        "name": block.name,
                        "arguments": json.dumps(block.arguments, ensure_ascii=False, separators=(",", ":")),
                        **namespace_field,
                    }
                )

    if not output:
        return
    messages.extend(output)


def _append_tool_result(
    messages: list[dict[str, Any]],
    msg: ToolResultMessage,
    model: Model,
    grammar_tool_input_properties: Mapping[str, str],
    deferred_tools: Mapping[str, Tool] | None,
    deferred_tools_mode: str | None,
    tool_options: Mapping[str, Any],
    loaded_tool_names: set[str],
) -> None:
    call_id = msg.tool_call_id.partition("|")[0]
    output = convert_tool_result_output(model, msg.content)

    item_type = (
        "custom_tool_call_output" if msg.tool_name in grammar_tool_input_properties else "function_call_output"
    )
    messages.append({"type": item_type, "call_id": call_id, "output": output})

    newly_loaded: list[Tool] = []
    for name in msg.added_tool_names or []:
        tool = (deferred_tools or {}).get(name)
        if tool is None or name in loaded_tool_names:
            continue
        loaded_tool_names.add(name)
        newly_loaded.append(tool)

    if not newly_loaded:
        return

    if deferred_tools_mode == "additional-tools":
        messages.append(
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": convert_responses_tools(newly_loaded, **tool_options),
            }
        )
    elif deferred_tools_mode == "tool-search":
        names = [tool.name for tool in newly_loaded]
        # 分两步写：f-string 里嵌套同种引号是 Python 3.12+ 语法，本包底线是 3.11。
        joined_names = ",".join(names)
        search_call_id = f"pi_tool_load_{short_hash(f'{msg.tool_call_id}:{joined_names}')}"
        messages.append(
            {
                "type": "tool_search_call",
                "call_id": search_call_id,
                "execution": "client",
                "status": "completed",
                "arguments": {"query": " ".join(names), "limit": len(names)},
            }
        )
        messages.append(
            {
                "type": "tool_search_output",
                "call_id": search_call_id,
                "execution": "client",
                "status": "completed",
                "tools": convert_responses_tools(newly_loaded, **{**tool_options, "defer_loading": True}),
            }
        )
