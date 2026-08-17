from dataclasses import replace

from pi_ai.api.openai_completions_compat import detect_compat
from pi_ai.api.openai_completions_messages import (
    apply_anthropic_cache_control,
    convert_messages,
    convert_tools,
    get_compat_cache_control,
    has_tool_history,
    normalize_tool_call_id,
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
from pi_ai.utils.hash import short_hash


def make_model(*, provider="acme", model_id="m", reasoning=False, vision=False) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider=provider,
        base_url="https://acme.test/v1",
        reasoning=reasoning,
        input=["text", "image"] if vision else ["text"],
        cost=ModelCost(),
        context_window=100000,
        max_tokens=4096,
    )


BASE_COMPAT = detect_compat(make_model())


def assistant(content, *, provider="acme", model_id="m", stop_reason="stop"):
    return AssistantMessage(
        content=content,
        api="openai-completions",
        provider=provider,
        model=model_id,
        stop_reason=stop_reason,
        timestamp=1,
    )


# --- 工具调用 id 归一化 ---


def test_pipe_ids_are_sanitized_and_joined():
    assert normalize_tool_call_id(make_model(), "call_abc|item_x") == "call_abc_item_x"


def test_pipe_ids_replace_unsafe_characters():
    assert normalize_tool_call_id(make_model(), "call+a/b|item=c") == "call_a_b_item_c"


def test_pipe_ids_without_item_part_keep_only_the_call_id():
    assert normalize_tool_call_id(make_model(), "call_abc|") == "call_abc"


def test_overlong_pipe_ids_fall_back_to_a_hashed_suffix():
    raw = "call_" + "a" * 40 + "|" + "b" * 400
    result = normalize_tool_call_id(make_model(), raw)
    assert len(result) <= 40
    assert result.endswith("_" + short_hash(raw)[:8])


def test_openai_provider_truncates_long_ids():
    assert len(normalize_tool_call_id(make_model(provider="openai"), "x" * 60)) == 40


def test_other_providers_keep_long_ids_untouched():
    assert normalize_tool_call_id(make_model(provider="acme"), "x" * 60) == "x" * 60


# --- 系统提示 ---


def test_system_prompt_uses_system_role_for_non_reasoning_models():
    ctx = Context(system_prompt="be brief", messages=[user_text("hi")])
    messages = convert_messages(make_model(), ctx, BASE_COMPAT)
    assert messages[0] == {"role": "system", "content": "be brief"}


def test_system_prompt_uses_developer_role_for_reasoning_models():
    model = make_model(reasoning=True)
    ctx = Context(system_prompt="be brief", messages=[user_text("hi")])
    messages = convert_messages(model, ctx, detect_compat(model))
    assert messages[0]["role"] == "developer"


def test_system_prompt_falls_back_to_system_when_developer_role_unsupported():
    model = make_model(reasoning=True)
    compat = replace(detect_compat(model), supports_developer_role=False)
    ctx = Context(system_prompt="be brief", messages=[user_text("hi")])
    assert convert_messages(model, ctx, compat)[0]["role"] == "system"


# --- user 消息 ---


def test_user_string_content():
    messages = convert_messages(make_model(), Context(messages=[user_text("hi")]), BASE_COMPAT)
    assert messages == [{"role": "user", "content": "hi"}]


def test_user_array_content_maps_text_and_images():
    model = make_model(vision=True)
    msg = UserMessage(
        content=[TextContent(text="look"), ImageContent(data="AAA", mime_type="image/png")],
        timestamp=1,
    )
    messages = convert_messages(model, Context(messages=[msg]), detect_compat(model))
    assert messages[0]["content"] == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]


def test_user_message_with_empty_content_array_is_skipped():
    msg = UserMessage(content=[], timestamp=1)
    assert convert_messages(make_model(), Context(messages=[msg]), BASE_COMPAT) == []


# --- assistant 消息 ---


def test_assistant_text_is_sent_as_a_plain_string():
    messages = convert_messages(
        make_model(), Context(messages=[assistant([TextContent(text="hello")])]), BASE_COMPAT
    )
    assert messages[0] == {"role": "assistant", "content": "hello"}


def test_assistant_thinking_goes_to_the_signature_named_field():
    msg = assistant(
        [ThinkingContent(thinking="deep", thinking_signature="reasoning_content"), TextContent(text="answer")]
    )
    messages = convert_messages(make_model(), Context(messages=[msg]), BASE_COMPAT)
    assert messages[0]["content"] == "answer"
    assert messages[0]["reasoning_content"] == "deep"


def test_opencode_go_reasoning_signature_is_renamed():
    model = make_model(provider="opencode-go")
    msg = assistant(
        [ThinkingContent(thinking="deep", thinking_signature="reasoning"), TextContent(text="answer")],
        provider="opencode-go",
    )
    messages = convert_messages(model, Context(messages=[msg]), detect_compat(model))
    assert messages[0]["reasoning_content"] == "deep"
    assert "reasoning" not in messages[0]


def test_thinking_only_assistant_messages_are_dropped():
    """只有思考块、既无文本也无工具调用的 assistant 回合会被整条丢弃。

    TS 侧同样如此：content 保持 None、没有 tool_calls，就走到末尾的跳过分支。
    签名字段虽然被写上了，但整条消息不会发出去。
    """
    msg = assistant([ThinkingContent(thinking="deep", thinking_signature="reasoning")])
    assert convert_messages(make_model(), Context(messages=[msg]), BASE_COMPAT) == []


def test_thinking_as_text_produces_content_parts():
    compat = replace(BASE_COMPAT, requires_thinking_as_text=True)
    msg = assistant([ThinkingContent(thinking="deep"), TextContent(text="answer")])
    messages = convert_messages(make_model(), Context(messages=[msg]), compat)
    assert messages[0]["content"] == [
        {"type": "text", "text": "deep"},
        {"type": "text", "text": "answer"},
    ]


def test_multiple_thinking_blocks_join_with_newline():
    msg = assistant(
        [
            ThinkingContent(thinking="one", thinking_signature="reasoning"),
            ThinkingContent(thinking="two", thinking_signature="reasoning"),
            TextContent(text="answer"),
        ]
    )
    messages = convert_messages(make_model(), Context(messages=[msg]), BASE_COMPAT)
    assert messages[0]["reasoning"] == "one\ntwo"


def test_assistant_tool_calls_are_serialized_as_functions():
    msg = assistant([ToolCall(id="c1", name="get_time", arguments={"tz": "UTC"})], stop_reason="toolUse")
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="c1", tool_name="get_time", timestamp=2)])
    messages = convert_messages(make_model(), ctx, BASE_COMPAT)
    assert messages[0]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "get_time", "arguments": '{"tz":"UTC"}'}}
    ]


def test_grammar_tool_calls_are_serialized_as_custom():
    msg = assistant([ToolCall(id="c1", name="patch", arguments={"input": "diff"})], stop_reason="toolUse")
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="c1", tool_name="patch", timestamp=2)])
    messages = convert_messages(make_model(), ctx, BASE_COMPAT, {"patch": "input"})
    assert messages[0]["tool_calls"] == [
        {"id": "c1", "type": "custom", "custom": {"name": "patch", "input": "diff"}}
    ]


def test_thought_signatures_become_reasoning_details():
    detail = '{"type":"reasoning.encrypted","id":"c1","data":"blob"}'
    msg = assistant(
        [ToolCall(id="c1", name="t", arguments={}, thought_signature=detail)], stop_reason="toolUse"
    )
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="c1", tool_name="t", timestamp=2)])
    messages = convert_messages(make_model(), ctx, BASE_COMPAT)
    assert messages[0]["reasoning_details"] == [
        {"type": "reasoning.encrypted", "id": "c1", "data": "blob"}
    ]


def test_unparseable_thought_signatures_are_dropped():
    msg = assistant(
        [ToolCall(id="c1", name="t", arguments={}, thought_signature="not json")], stop_reason="toolUse"
    )
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="c1", tool_name="t", timestamp=2)])
    messages = convert_messages(make_model(), ctx, BASE_COMPAT)
    assert "reasoning_details" not in messages[0]


def test_reasoning_content_is_backfilled_when_required():
    model = make_model(reasoning=True)
    compat = replace(detect_compat(model), requires_reasoning_content_on_assistant_messages=True)
    msg = assistant([TextContent(text="hi")], model_id=model.id)
    messages = convert_messages(model, Context(messages=[msg]), compat)
    assert messages[0]["reasoning_content"] == ""


def test_empty_assistant_messages_are_skipped():
    msg = assistant([])
    assert convert_messages(make_model(), Context(messages=[msg]), BASE_COMPAT) == []


def test_assistant_content_is_empty_string_when_required():
    compat = replace(BASE_COMPAT, requires_assistant_after_tool_result=True)
    msg = assistant([ToolCall(id="c1", name="t", arguments={})], stop_reason="toolUse")
    ctx = Context(messages=[msg, ToolResultMessage(tool_call_id="c1", tool_name="t", timestamp=2)])
    messages = convert_messages(make_model(), ctx, compat)
    assert messages[0]["content"] == ""


# --- 工具结果 ---


def test_tool_result_text_is_joined_with_newlines():
    msg = ToolResultMessage(
        tool_call_id="c1",
        tool_name="t",
        content=[TextContent(text="line1"), TextContent(text="line2")],
        timestamp=1,
    )
    messages = convert_messages(make_model(), Context(messages=[msg]), BASE_COMPAT)
    assert messages[0] == {"role": "tool", "content": "line1\nline2", "tool_call_id": "c1"}


def test_tool_result_without_output_uses_a_placeholder():
    msg = ToolResultMessage(tool_call_id="c1", tool_name="t", content=[], timestamp=1)
    messages = convert_messages(make_model(), Context(messages=[msg]), BASE_COMPAT)
    assert messages[0]["content"] == "(no tool output)"


def test_tool_result_with_only_images_uses_the_image_placeholder():
    model = make_model(vision=True)
    msg = ToolResultMessage(
        tool_call_id="c1",
        tool_name="t",
        content=[ImageContent(data="AAA", mime_type="image/png")],
        timestamp=1,
    )
    messages = convert_messages(model, Context(messages=[msg]), detect_compat(model))
    assert messages[0]["content"] == "(see attached image)"


def test_tool_result_name_is_added_when_required():
    compat = replace(BASE_COMPAT, requires_tool_result_name=True)
    msg = ToolResultMessage(
        tool_call_id="c1", tool_name="get_time", content=[TextContent(text="ok")], timestamp=1
    )
    messages = convert_messages(make_model(), Context(messages=[msg]), compat)
    assert messages[0]["name"] == "get_time"


def test_tool_result_images_become_a_follow_up_user_message():
    model = make_model(vision=True)
    msg = ToolResultMessage(
        tool_call_id="c1",
        tool_name="t",
        content=[TextContent(text="chart"), ImageContent(data="AAA", mime_type="image/png")],
        timestamp=1,
    )
    messages = convert_messages(model, Context(messages=[msg]), detect_compat(model))
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["text"] == "Attached image(s) from tool result:"
    assert messages[1]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAA"


def test_consecutive_tool_results_are_merged_into_one_run():
    messages = convert_messages(
        make_model(),
        Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="c1", tool_name="t", content=[TextContent(text="a")], timestamp=1
                ),
                ToolResultMessage(
                    tool_call_id="c2", tool_name="t", content=[TextContent(text="b")], timestamp=2
                ),
                user_text("next"),
            ]
        ),
        BASE_COMPAT,
    )
    assert [m["role"] for m in messages] == ["tool", "tool", "user"]


def test_synthetic_assistant_bridges_tool_result_and_user():
    compat = replace(BASE_COMPAT, requires_assistant_after_tool_result=True)
    messages = convert_messages(
        make_model(),
        Context(
            messages=[
                ToolResultMessage(
                    tool_call_id="c1", tool_name="t", content=[TextContent(text="a")], timestamp=1
                ),
                user_text("next"),
            ]
        ),
        compat,
    )
    assert [m["role"] for m in messages] == ["tool", "assistant", "user"]
    assert messages[1]["content"] == "I have processed the tool results."


def test_kimi_deferred_tools_emit_a_system_tools_message():
    compat = replace(BASE_COMPAT, deferred_tools_mode="kimi")
    tool = Tool(name="later", description="d", parameters={"type": "object"})
    ctx = Context(
        messages=[
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="t",
                content=[TextContent(text="ok")],
                added_tool_names=["later"],
                timestamp=1,
            )
        ],
        tools=[tool],
    )
    messages = convert_messages(make_model(), ctx, compat)
    assert messages[-1]["role"] == "system"
    assert messages[-1]["tools"][0]["function"]["name"] == "later"


# --- 工具序列化 ---


def test_function_tools_include_strict_when_supported():
    tool = Tool(name="t", description="d", parameters={"type": "object"})
    converted = convert_tools([tool], BASE_COMPAT)
    assert converted[0]["function"]["strict"] is False


def test_strict_is_omitted_entirely_when_unsupported():
    compat = replace(BASE_COMPAT, supports_strict_mode=False)
    tool = Tool(name="t", description="d", parameters={"type": "object"})
    assert "strict" not in convert_tools([tool], compat)[0]["function"]


def test_json_schema_constrained_tools_enable_strict():
    tool = Tool(
        name="t",
        description="d",
        parameters={"type": "object"},
        constrained_sampling={"type": "json_schema", "strict": "prefer"},
    )
    assert convert_tools([tool], BASE_COMPAT)[0]["function"]["strict"] is True


def test_grammar_tools_become_custom_tools():
    compat = replace(BASE_COMPAT, supports_openai_grammar_tools=True)
    tool = Tool(
        name="patch",
        description="d",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
        constrained_sampling={"type": "grammar", "variants": {"openai_lark": "start: /.+/s"}},
    )
    converted = convert_tools([tool], compat)
    assert converted[0]["type"] == "custom"
    assert converted[0]["custom"]["format"]["grammar"] == {"syntax": "lark", "definition": "start: /.+/s"}


# --- 缓存标记 ---


def test_cache_control_is_disabled_for_non_anthropic_formats():
    assert get_compat_cache_control(BASE_COMPAT, "short") is None


def test_cache_control_short_and_long_retention():
    compat = replace(BASE_COMPAT, cache_control_format="anthropic")
    assert get_compat_cache_control(compat, "none") is None
    assert get_compat_cache_control(compat, "short") == {"type": "ephemeral"}
    assert get_compat_cache_control(compat, "long") == {"type": "ephemeral", "ttl": "1h"}

    without_long = replace(compat, supports_long_cache_retention=False)
    assert get_compat_cache_control(without_long, "long") == {"type": "ephemeral"}


def test_cache_control_lands_on_system_last_tool_and_last_message():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ]
    tools = [{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]
    control = {"type": "ephemeral"}
    apply_anthropic_cache_control(messages, tools, control)

    assert messages[0]["content"] == [{"type": "text", "text": "sys", "cache_control": control}]
    assert tools[1]["cache_control"] == control
    assert tools[0].get("cache_control") is None
    assert messages[2]["content"] == [{"type": "text", "text": "there", "cache_control": control}]
    assert messages[1]["content"] == "hi"


def test_cache_control_skips_empty_messages_and_walks_backwards():
    messages = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": ""},
    ]
    control = {"type": "ephemeral"}
    apply_anthropic_cache_control(messages, None, control)
    assert messages[0]["content"] == [{"type": "text", "text": "earlier", "cache_control": control}]


def test_cache_control_marks_the_last_text_part_of_an_array():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "b"},
            ],
        }
    ]
    control = {"type": "ephemeral"}
    apply_anthropic_cache_control(messages, None, control)
    assert messages[0]["content"][2]["cache_control"] == control
    assert "cache_control" not in messages[0]["content"][0]


# --- 其它 ---


def test_has_tool_history_detects_calls_and_results():
    assert has_tool_history([user_text("hi")]) is False
    assert has_tool_history([ToolResultMessage(tool_call_id="c", tool_name="t")]) is True
    assert has_tool_history([assistant([ToolCall(id="c", name="t", arguments={})])]) is True
