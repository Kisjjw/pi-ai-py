from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pi_ai.types import GrammarSampling, JsonSchemaSampling, Tool


@dataclass(frozen=True)
class GrammarConstrainedSampling:
    format: str  # "lark" | "regex"
    definition: str
    input_property: str


@dataclass
class GrammarToolInputJsonBuffer:
    """把语法工具的纯字符串输入包装成 JSON 增量时的累积状态。"""

    input: str = ""
    started: bool = False
    closed: bool = False


def get_grammar_tool_input(tool_name: str, arguments: dict[str, Any], input_property: str) -> str:
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise ValueError(f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.')
    return value


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer,
    input_property: str,
    next_input: str,
    close: bool,
) -> str | None:
    """把语法工具的输入增量转成等价的 JSON 文本增量。

    上层协议的工具参数增量是 JSON 片段，而语法工具的输入是裸字符串，
    因此这里要现场拼出 `{"prop":"...` 的形状。
    """
    if buffer.closed:
        if close and next_input == buffer.input:
            return None
        raise ValueError(f'grammar tool input for property "{input_property}" changed after it was closed')

    if not next_input.startswith(buffer.input):
        raise ValueError(f'grammar tool input for property "{input_property}" changed non-monotonically')

    input_delta = next_input[len(buffer.input) :]
    if not close and not input_delta:
        return None

    delta = ""
    if not buffer.started:
        delta += "{" + json.dumps(input_property, ensure_ascii=False) + ':"'
        buffer.started = True

    # ensure_ascii=False 是必须的：否则非 ASCII 会被转义成 \uXXXX，与 TS 产出不同。
    delta += json.dumps(input_delta, ensure_ascii=False)[1:-1]
    buffer.input = next_input

    if close:
        delta += '"}'
        buffer.closed = True

    return delta


def _infer_grammar_input_property(tool: Tool) -> str:
    schema = tool.parameters or {}
    if schema.get("type") != "object":
        raise ValueError("grammar constrained sampling requires an object parameter schema")

    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(required[0], str):
        raise ValueError("grammar constrained sampling requires exactly one required string property")

    input_property = required[0]
    properties = schema.get("properties") or {}
    if not properties.get(input_property):
        raise ValueError(f"grammar constrained sampling requires a properties entry for {input_property}")
    if properties[input_property].get("type") != "string":
        raise ValueError(f"grammar constrained sampling property {input_property} must have type string")
    return input_property


def _json_schema_config(tool: Tool) -> JsonSchemaSampling | None:
    config = tool.constrained_sampling
    return config if isinstance(config, JsonSchemaSampling) else None


def _grammar_config(tool: Tool) -> GrammarSampling | None:
    config = tool.constrained_sampling
    return config if isinstance(config, GrammarSampling) else None


def resolve_json_schema_strict_sampling(tool: Tool, supports_strict_mode: bool) -> bool | None:
    """决定是否给该工具打开厂商侧的严格 JSON Schema 约束。"""
    config = _json_schema_config(tool)
    if config is None:
        return None

    if supports_strict_mode:
        return True

    if config.strict == "require":
        raise ValueError(
            f'Tool "{tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.'
        )
    return None


def resolve_grammar_constrained_sampling(
    tool: Tool,
    supports_openai_grammar_tools: bool,
) -> GrammarConstrainedSampling | None:
    config = _grammar_config(tool)
    if config is None:
        return None
    if not supports_openai_grammar_tools:
        return None

    variants = config.variants or {}
    lark = variants.get("openai_lark")
    regex = variants.get("openai_regex")
    has_lark = isinstance(lark, str) and lark.strip() != ""
    has_regex = isinstance(regex, str) and regex.strip() != ""

    if not has_lark and not has_regex:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: no supported grammar variant was provided.'
        )

    try:
        input_property = _infer_grammar_input_property(tool)
    except ValueError as error:
        raise ValueError(f'Tool "{tool.name}" cannot use grammar constrained sampling: {error}.') from error

    return GrammarConstrainedSampling(
        format="lark" if has_lark else "regex",
        definition=lark if has_lark else regex,  # type: ignore[arg-type]
        input_property=input_property,
    )


def create_grammar_tool_input_properties(
    tools: Sequence[Tool] | None,
    supports_openai_grammar_tools: bool,
) -> dict[str, str]:
    properties: dict[str, str] = {}
    for tool in tools or []:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar is not None:
            properties[tool.name] = grammar.input_property
    return properties
