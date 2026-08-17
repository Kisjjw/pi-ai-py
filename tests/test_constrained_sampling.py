import pytest

from pi_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    get_grammar_tool_input,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from pi_ai.api.github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
    infer_copilot_initiator,
)
from pi_ai.api.openai_prompt_cache import clamp_openai_prompt_cache_key
from pi_ai.types import ImageContent, TextContent, Tool, ToolResultMessage, user_text

GRAMMAR_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
    "additionalProperties": False,
}


def grammar_tool(**overrides) -> Tool:
    defaults = dict(
        name="apply_patch",
        description="Apply a patch",
        parameters=GRAMMAR_SCHEMA,
        constrained_sampling={"type": "grammar", "variants": {"openai_lark": "start: /.+/s"}},
    )
    defaults.update(overrides)
    return Tool(**defaults)


def json_tool(strict="prefer") -> Tool:
    return Tool(
        name="edit_file",
        description="Edit",
        parameters={"type": "object", "properties": {}},
        constrained_sampling={"type": "json_schema", "strict": strict},
    )


def test_strict_sampling_returns_none_without_config():
    assert resolve_json_schema_strict_sampling(Tool(name="t"), True) is None


def test_strict_sampling_returns_none_for_grammar_config():
    assert resolve_json_schema_strict_sampling(grammar_tool(), True) is None


def test_strict_sampling_enabled_when_supported():
    assert resolve_json_schema_strict_sampling(json_tool("prefer"), True) is True


def test_strict_prefer_falls_back_when_unsupported():
    assert resolve_json_schema_strict_sampling(json_tool("prefer"), False) is None


def test_strict_require_raises_when_unsupported():
    with pytest.raises(ValueError, match="requires JSON-schema constrained sampling"):
        resolve_json_schema_strict_sampling(json_tool("require"), False)


def test_grammar_sampling_returns_none_when_unsupported():
    assert resolve_grammar_constrained_sampling(grammar_tool(), False) is None


def test_grammar_sampling_prefers_lark_over_regex():
    tool = grammar_tool(
        constrained_sampling={
            "type": "grammar",
            "variants": {"openai_lark": "lark-def", "openai_regex": "regex-def"},
        }
    )
    grammar = resolve_grammar_constrained_sampling(tool, True)
    assert grammar is not None
    assert grammar.format == "lark"
    assert grammar.definition == "lark-def"
    assert grammar.input_property == "input"


def test_grammar_sampling_falls_back_to_regex():
    tool = grammar_tool(constrained_sampling={"type": "grammar", "variants": {"openai_regex": "re"}})
    grammar = resolve_grammar_constrained_sampling(tool, True)
    assert grammar is not None
    assert grammar.format == "regex"


def test_grammar_sampling_rejects_empty_variants():
    tool = grammar_tool(constrained_sampling={"type": "grammar", "variants": {"openai_lark": "   "}})
    with pytest.raises(ValueError, match="no supported grammar variant"):
        resolve_grammar_constrained_sampling(tool, True)


def test_grammar_sampling_rejects_non_object_schema():
    tool = grammar_tool(parameters={"type": "string"})
    with pytest.raises(ValueError, match="object parameter schema"):
        resolve_grammar_constrained_sampling(tool, True)


def test_grammar_sampling_rejects_multiple_required_properties():
    tool = grammar_tool(
        parameters={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        }
    )
    with pytest.raises(ValueError, match="exactly one required string property"):
        resolve_grammar_constrained_sampling(tool, True)


def test_grammar_sampling_rejects_missing_property_entry():
    tool = grammar_tool(parameters={"type": "object", "properties": {}, "required": ["input"]})
    with pytest.raises(ValueError, match="properties entry"):
        resolve_grammar_constrained_sampling(tool, True)


def test_grammar_sampling_rejects_non_string_property():
    tool = grammar_tool(
        parameters={"type": "object", "properties": {"input": {"type": "number"}}, "required": ["input"]}
    )
    with pytest.raises(ValueError, match="must have type string"):
        resolve_grammar_constrained_sampling(tool, True)


def test_input_property_map_only_includes_grammar_tools():
    tools = [grammar_tool(), json_tool(), Tool(name="plain")]
    assert create_grammar_tool_input_properties(tools, True) == {"apply_patch": "input"}
    assert create_grammar_tool_input_properties(tools, False) == {}


def test_get_grammar_tool_input_requires_a_string():
    assert get_grammar_tool_input("t", {"input": "x"}, "input") == "x"
    with pytest.raises(ValueError, match="to be a string"):
        get_grammar_tool_input("t", {"input": 1}, "input")


def test_json_delta_opens_the_object_on_first_chunk():
    buffer = GrammarToolInputJsonBuffer()
    assert append_grammar_tool_input_json_delta(buffer, "input", "ab", False) == '{"input":"ab'


def test_json_delta_only_emits_the_new_suffix():
    buffer = GrammarToolInputJsonBuffer()
    append_grammar_tool_input_json_delta(buffer, "input", "ab", False)
    assert append_grammar_tool_input_json_delta(buffer, "input", "abcd", False) == "cd"


def test_json_delta_closes_the_object():
    buffer = GrammarToolInputJsonBuffer()
    append_grammar_tool_input_json_delta(buffer, "input", "ab", False)
    assert append_grammar_tool_input_json_delta(buffer, "input", "ab", True) == '"}'


def test_json_delta_escapes_control_characters():
    buffer = GrammarToolInputJsonBuffer()
    delta = append_grammar_tool_input_json_delta(buffer, "input", 'a"\nb', False)
    assert delta == '{"input":"a\\"\\nb'


def test_json_delta_keeps_non_ascii_literal():
    buffer = GrammarToolInputJsonBuffer()
    delta = append_grammar_tool_input_json_delta(buffer, "input", "中文", False)
    assert delta == '{"input":"中文'


def test_json_delta_skips_empty_non_closing_updates():
    buffer = GrammarToolInputJsonBuffer()
    append_grammar_tool_input_json_delta(buffer, "input", "ab", False)
    assert append_grammar_tool_input_json_delta(buffer, "input", "ab", False) is None


def test_json_delta_rejects_non_monotonic_input():
    buffer = GrammarToolInputJsonBuffer()
    append_grammar_tool_input_json_delta(buffer, "input", "abc", False)
    with pytest.raises(ValueError, match="non-monotonically"):
        append_grammar_tool_input_json_delta(buffer, "input", "xyz", False)


def test_json_delta_after_close_is_idempotent_then_rejects():
    buffer = GrammarToolInputJsonBuffer()
    append_grammar_tool_input_json_delta(buffer, "input", "ab", True)
    assert append_grammar_tool_input_json_delta(buffer, "input", "ab", True) is None
    with pytest.raises(ValueError, match="after it was closed"):
        append_grammar_tool_input_json_delta(buffer, "input", "abc", False)


def test_prompt_cache_key_is_clamped_to_64_code_points():
    assert clamp_openai_prompt_cache_key(None) is None
    assert clamp_openai_prompt_cache_key("short") == "short"
    assert clamp_openai_prompt_cache_key("x" * 100) == "x" * 64
    assert clamp_openai_prompt_cache_key("中" * 100) == "中" * 64


def test_copilot_initiator_is_agent_after_non_user_messages():
    assert infer_copilot_initiator([user_text("hi")]) == "user"
    assert infer_copilot_initiator([ToolResultMessage(tool_call_id="c", tool_name="t")]) == "agent"
    assert infer_copilot_initiator([]) == "user"


def test_copilot_vision_detects_images_in_user_and_tool_results():
    assert has_copilot_vision_input([user_text("hi")]) is False
    from pi_ai.types import UserMessage

    with_image = UserMessage(content=[ImageContent(data="x", mime_type="image/png")])
    assert has_copilot_vision_input([with_image]) is True

    tool_image = ToolResultMessage(
        tool_call_id="c",
        tool_name="t",
        content=[TextContent(text="a"), ImageContent(data="x", mime_type="image/png")],
    )
    assert has_copilot_vision_input([tool_image]) is True


def test_copilot_dynamic_headers():
    headers = build_copilot_dynamic_headers([user_text("hi")], has_images=False)
    assert headers == {"X-Initiator": "user", "Openai-Intent": "conversation-edits"}

    headers = build_copilot_dynamic_headers([user_text("hi")], has_images=True)
    assert headers["Copilot-Vision-Request"] == "true"
