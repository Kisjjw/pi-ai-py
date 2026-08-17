import json

from pi_ai.api.openai_responses_shared import (
    build_foreign_responses_item_id,
    convert_responses_messages,
    convert_responses_tools,
    convert_tool_result_output,
    encode_text_signature_v1,
    normalize_id_part,
    parse_text_signature,
)
from pi_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    user_text,
)

OPENAI_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})


def make_model(*, provider="openai", model_id="gpt-5", reasoning=True, vision=False, compat=None) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-responses",
        provider=provider,
        base_url="https://api.openai.com/v1",
        reasoning=reasoning,
        input=["text", "image"] if vision else ["text"],
        cost=ModelCost(),
        context_window=200000,
        max_tokens=32000,
        compat=compat,
    )


def assistant(content, *, provider="openai", model_id="gpt-5", api="openai-responses", stop_reason="stop"):
    return AssistantMessage(
        content=content,
        api=api,
        provider=provider,
        model=model_id,
        stop_reason=stop_reason,
        timestamp=1,
    )


# --- 文本签名 ---


def test_text_signature_round_trip():
    assert parse_text_signature(encode_text_signature_v1("msg_1")) == {"id": "msg_1"}
    assert parse_text_signature(encode_text_signature_v1("msg_1", "final_answer")) == {
        "id": "msg_1",
        "phase": "final_answer",
    }


def test_legacy_plain_string_signatures_still_parse():
    assert parse_text_signature("msg_legacy") == {"id": "msg_legacy"}


def test_malformed_json_signature_falls_back_to_the_raw_string():
    assert parse_text_signature("{not json") == {"id": "{not json"}


def test_unknown_phase_is_dropped():
    raw = json.dumps({"v": 1, "id": "m", "phase": "weird"})
    assert parse_text_signature(raw) == {"id": "m"}


def test_empty_signature_yields_none():
    assert parse_text_signature(None) is None
    assert parse_text_signature("") is None


# --- id 归一化 ---


def test_id_parts_are_sanitized_truncated_and_trimmed():
    assert normalize_id_part("call+a/b") == "call_a_b"
    assert len(normalize_id_part("x" * 100)) == 64
    # 截断后末尾的下划线要去掉
    assert normalize_id_part("a" * 63 + "+b") == "a" * 63


def test_foreign_item_ids_are_hashed_with_an_fc_prefix():
    result = build_foreign_responses_item_id("rs_verylongid")
    assert result.startswith("fc_")
    assert len(result) <= 64


def test_composite_ids_survive_for_allowed_providers():
    msg = assistant([ToolCall(id="call_1|fc_abc", name="t", arguments={})], stop_reason="toolUse")
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="call_1|fc_abc", tool_name="t", timestamp=2)])
    items = convert_responses_messages(make_model(), ctx, OPENAI_PROVIDERS)
    assert items[0]["call_id"] == "call_1"
    assert items[0]["id"] == "fc_abc"


def test_same_model_tool_call_ids_are_never_rewritten():
    """id 归一化只在跨模型回放时触发，同模型的原样保留。"""
    msg = assistant([ToolCall(id="call_1|fc_abc", name="t", arguments={})], stop_reason="toolUse")
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="call_1|fc_abc", tool_name="t", timestamp=2)])
    items = convert_responses_messages(make_model(), ctx, OPENAI_PROVIDERS)
    assert items[0]["call_id"] == "call_1"
    assert items[0]["id"] == "fc_abc"


def test_composite_ids_are_flattened_for_other_providers():
    """非白名单 provider 跨模型回放时整段压平，不再保留 | 结构。"""
    model = make_model(provider="xai")
    msg = assistant(
        [ToolCall(id="call_1|fc_abc", name="t", arguments={})],
        provider="xai",
        model_id="grok-old",
        stop_reason="toolUse",
    )
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="call_1|fc_abc", tool_name="t", timestamp=2)])
    items = convert_responses_messages(model, ctx, OPENAI_PROVIDERS)
    assert items[0]["call_id"] == "call_1_fc_abc"
    assert items[0]["id"] is None


def test_foreign_tool_calls_get_a_hashed_item_id():
    msg = assistant(
        [ToolCall(id="call_1|rs_foreign", name="t", arguments={})],
        provider="anthropic",
        api="anthropic-messages",
        stop_reason="toolUse",
    )
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="call_1|rs_foreign", tool_name="t", timestamp=2)])
    items = convert_responses_messages(make_model(), ctx, OPENAI_PROVIDERS)
    # 跨 provider 时 item id 被丢掉（因为不是 fc_ 前缀会被置空的规则不适用，
    # 这里生成的是 fc_ 哈希，但同 provider 判定为 false，所以 id 保留）
    assert items[0]["call_id"] == "call_1"
    assert items[0]["id"] is not None
    assert items[0]["id"].startswith("fc_")


def test_item_id_is_dropped_when_the_model_changed():
    msg = assistant([ToolCall(id="call_1|fc_abc", name="t", arguments={})], model_id="gpt-4o", stop_reason="toolUse")
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="call_1|fc_abc", tool_name="t", timestamp=2)])
    items = convert_responses_messages(make_model(model_id="gpt-5"), ctx, OPENAI_PROVIDERS)
    assert items[0]["id"] is None


def test_non_fc_item_ids_are_dropped_for_function_calls():
    msg = assistant([ToolCall(id="call_1|ctc_abc", name="t", arguments={})], stop_reason="toolUse")
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="call_1|ctc_abc", tool_name="t", timestamp=2)])
    items = convert_responses_messages(make_model(), ctx, OPENAI_PROVIDERS)
    assert items[0]["type"] == "function_call"
    assert items[0]["id"] is None


# --- system 与 user ---


def test_system_prompt_uses_developer_role_for_reasoning_models():
    ctx = Context(system_prompt="be brief", messages=[user_text("hi")])
    items = convert_responses_messages(make_model(), ctx, OPENAI_PROVIDERS)
    assert items[0] == {"role": "developer", "content": "be brief"}


def test_system_prompt_uses_system_role_for_non_reasoning_models():
    ctx = Context(system_prompt="be brief", messages=[user_text("hi")])
    items = convert_responses_messages(make_model(reasoning=False), ctx, OPENAI_PROVIDERS)
    assert items[0]["role"] == "system"


def test_developer_role_can_be_disabled_by_compat():
    model = make_model(compat={"supportsDeveloperRole": False})
    ctx = Context(system_prompt="be brief", messages=[user_text("hi")])
    assert convert_responses_messages(model, ctx, OPENAI_PROVIDERS)[0]["role"] == "system"


def test_system_prompt_can_be_excluded():
    ctx = Context(system_prompt="be brief", messages=[user_text("hi")])
    items = convert_responses_messages(make_model(), ctx, OPENAI_PROVIDERS, include_system_prompt=False)
    assert items[0]["role"] == "user"


def test_user_string_becomes_an_input_text_block():
    items = convert_responses_messages(make_model(), Context(messages=[user_text("hi")]), OPENAI_PROVIDERS)
    assert items[0] == {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}


def test_user_images_become_input_image_blocks():
    msg = UserMessage(
        content=[TextContent(text="look"), ImageContent(data="AAA", mime_type="image/png")], timestamp=1
    )
    items = convert_responses_messages(make_model(vision=True), Context(messages=[msg]), OPENAI_PROVIDERS)
    assert items[0]["content"] == [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,AAA"},
    ]


# --- assistant ---


def test_thinking_signature_is_replayed_as_a_reasoning_item():
    reasoning_item = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "blob"}
    msg = assistant([ThinkingContent(thinking="x", thinking_signature=json.dumps(reasoning_item))])
    items = convert_responses_messages(make_model(), Context(messages=[msg]), OPENAI_PROVIDERS)
    assert items[0] == reasoning_item


def test_thinking_without_a_signature_is_dropped():
    msg = assistant([ThinkingContent(thinking="deep"), TextContent(text="answer")])
    items = convert_responses_messages(make_model(), Context(messages=[msg]), OPENAI_PROVIDERS)
    assert [i["type"] for i in items] == ["message"]


def test_assistant_text_becomes_a_message_item():
    msg = assistant([TextContent(text="hello", text_signature=encode_text_signature_v1("msg_x"))])
    items = convert_responses_messages(make_model(), Context(messages=[msg]), OPENAI_PROVIDERS)
    assert items[0]["type"] == "message"
    assert items[0]["id"] == "msg_x"
    assert items[0]["content"][0] == {"type": "output_text", "text": "hello", "annotations": []}


def test_text_without_a_signature_gets_a_synthetic_id():
    msg = assistant([TextContent(text="a"), TextContent(text="b")])
    items = convert_responses_messages(make_model(), Context(messages=[msg]), OPENAI_PROVIDERS)
    assert items[0]["id"] == "msg_pi_0"
    assert items[1]["id"] == "msg_pi_0_1"


def test_overlong_text_ids_are_hashed():
    msg = assistant([TextContent(text="a", text_signature=encode_text_signature_v1("m" * 100))])
    items = convert_responses_messages(make_model(), Context(messages=[msg]), OPENAI_PROVIDERS)
    assert items[0]["id"].startswith("msg_")
    assert len(items[0]["id"]) <= 64


def test_text_phase_is_preserved():
    msg = assistant([TextContent(text="a", text_signature=encode_text_signature_v1("m", "final_answer"))])
    items = convert_responses_messages(make_model(), Context(messages=[msg]), OPENAI_PROVIDERS)
    assert items[0]["phase"] == "final_answer"


def test_tool_call_arguments_are_serialized():
    msg = assistant([ToolCall(id="call_1|fc_a", name="t", arguments={"x": 1})], stop_reason="toolUse")
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="call_1|fc_a", tool_name="t", timestamp=2)])
    items = convert_responses_messages(make_model(), ctx, OPENAI_PROVIDERS)
    assert items[0]["arguments"] == '{"x":1}'


def test_grammar_tool_calls_become_custom_tool_calls():
    msg = assistant([ToolCall(id="call_1|ctc_a", name="patch", arguments={"input": "diff"})], stop_reason="toolUse")
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="call_1|ctc_a", tool_name="patch", timestamp=2)])
    items = convert_responses_messages(
        make_model(), ctx, OPENAI_PROVIDERS, grammar_tool_input_properties={"patch": "input"}
    )
    assert items[0]["type"] == "custom_tool_call"
    assert items[0]["input"] == "diff"


def test_namespace_is_replayed_only_for_the_same_model():
    call = ToolCall(id="call_1|fc_a", name="t", arguments={}, namespace="ns")
    same = assistant([call], stop_reason="toolUse")
    ctx = Context(messages=[same, ToolResultMessage(tool_call_id="call_1|fc_a", tool_name="t", timestamp=2)])
    assert convert_responses_messages(make_model(), ctx, OPENAI_PROVIDERS)[0]["namespace"] == "ns"

    other = assistant([call], model_id="gpt-4o", stop_reason="toolUse")
    ctx = Context(messages=[other, ToolResultMessage(tool_call_id="call_1|fc_a", tool_name="t", timestamp=2)])
    assert "namespace" not in convert_responses_messages(make_model(), ctx, OPENAI_PROVIDERS)[0]


def test_assistant_with_no_items_is_skipped():
    assert convert_responses_messages(make_model(), Context(messages=[assistant([])]), OPENAI_PROVIDERS) == []


# --- 工具结果 ---


def test_tool_result_text_output():
    msg = ToolResultMessage(
        tool_call_id="call_1|fc_a", tool_name="t", content=[TextContent(text="ok")], timestamp=1
    )
    items = convert_responses_messages(make_model(), Context(messages=[msg]), OPENAI_PROVIDERS)
    assert items[0] == {"type": "function_call_output", "call_id": "call_1", "output": "ok"}


def test_grammar_tool_results_use_the_custom_output_type():
    msg = ToolResultMessage(
        tool_call_id="call_1", tool_name="patch", content=[TextContent(text="ok")], timestamp=1
    )
    items = convert_responses_messages(
        make_model(), Context(messages=[msg]), OPENAI_PROVIDERS, grammar_tool_input_properties={"patch": "in"}
    )
    assert items[0]["type"] == "custom_tool_call_output"


def test_tool_result_output_placeholders():
    model = make_model()
    assert convert_tool_result_output(model, []) == "(no tool output)"
    assert convert_tool_result_output(model, [ImageContent(data="A", mime_type="image/png")]) == (
        "(see attached image)"
    )


def test_tool_result_images_are_kept_for_vision_models():
    output = convert_tool_result_output(
        make_model(vision=True),
        [TextContent(text="chart"), ImageContent(data="AAA", mime_type="image/png")],
    )
    assert output == [
        {"type": "input_text", "text": "chart"},
        {"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,AAA"},
    ]


# --- 延迟工具 ---


def deferred_tool(name="later") -> Tool:
    return Tool(name=name, description="d", parameters={"type": "object"})


def test_additional_tools_mode_emits_a_developer_item():
    msg = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="t",
        content=[TextContent(text="ok")],
        added_tool_names=["later"],
        timestamp=1,
    )
    items = convert_responses_messages(
        make_model(),
        Context(messages=[msg]),
        OPENAI_PROVIDERS,
        deferred_tools={"later": deferred_tool()},
        deferred_tools_mode="additional-tools",
    )
    assert items[1]["type"] == "additional_tools"
    assert items[1]["role"] == "developer"
    assert items[1]["tools"][0]["name"] == "later"


def test_tool_search_mode_emits_a_call_and_output_pair():
    msg = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="t",
        content=[TextContent(text="ok")],
        added_tool_names=["later"],
        timestamp=1,
    )
    items = convert_responses_messages(
        make_model(),
        Context(messages=[msg]),
        OPENAI_PROVIDERS,
        deferred_tools={"later": deferred_tool()},
        deferred_tools_mode="tool-search",
    )
    assert items[1]["type"] == "tool_search_call"
    assert items[2]["type"] == "tool_search_output"
    assert items[1]["call_id"] == items[2]["call_id"]
    assert items[2]["tools"][0]["defer_loading"] is True


def test_a_deferred_tool_is_only_loaded_once():
    messages = [
        ToolResultMessage(
            tool_call_id="c1", tool_name="t", added_tool_names=["later"], timestamp=1
        ),
        ToolResultMessage(
            tool_call_id="c2", tool_name="t", added_tool_names=["later"], timestamp=2
        ),
    ]
    items = convert_responses_messages(
        make_model(),
        Context(messages=messages),
        OPENAI_PROVIDERS,
        deferred_tools={"later": deferred_tool()},
        deferred_tools_mode="additional-tools",
    )
    assert sum(1 for i in items if i["type"] == "additional_tools") == 1


# --- 工具转换 ---


def make_tool(**overrides) -> Tool:
    defaults = dict(name="get_time", description="d", parameters={"type": "object", "properties": {}})
    defaults.update(overrides)
    return Tool(**defaults)


def test_function_tools_include_strict_when_supported():
    converted = convert_responses_tools([make_tool()], supports_strict_mode=True)
    assert converted[0] == {
        "type": "function",
        "name": "get_time",
        "description": "d",
        "parameters": {"type": "object", "properties": {}},
        "strict": False,
    }


def test_strict_is_omitted_when_unsupported():
    converted = convert_responses_tools([make_tool()], supports_strict_mode=False)
    assert "strict" not in converted[0]


def test_constrained_tools_turn_strict_on():
    tool = make_tool(constrained_sampling={"type": "json_schema", "strict": "prefer"})
    assert convert_responses_tools([tool], supports_strict_mode=True)[0]["strict"] is True


def test_grammar_tools_use_a_flat_format_object():
    """Responses 的语法工具 format 是平铺的，与 Chat Completions 的嵌套结构不同。"""
    tool = make_tool(
        name="patch",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
        constrained_sampling={"type": "grammar", "variants": {"openai_lark": "start: /.+/s"}},
    )
    converted = convert_responses_tools([tool], supports_openai_grammar_tools=True)
    assert converted[0]["format"] == {
        "type": "grammar",
        "syntax": "lark",
        "definition": "start: /.+/s",
    }


def test_defer_loading_is_marked():
    converted = convert_responses_tools([make_tool()], defer_loading=True)
    assert converted[0]["defer_loading"] is True
