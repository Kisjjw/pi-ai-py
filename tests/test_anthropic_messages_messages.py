from pi_ai.api.anthropic_messages_messages import (
    convert_content_blocks,
    convert_messages,
    convert_tool_result,
    convert_tools,
    from_claude_code_name,
    normalize_tool_call_id,
    to_claude_code_name,
)
from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    user_text,
)

CACHE = {"type": "ephemeral"}


def assistant(content, stop_reason="stop"):
    return AssistantMessage(
        content=content,
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        stop_reason=stop_reason,
        timestamp=1,
    )


# --- Claude Code 名字映射 ---


def test_tool_names_map_to_claude_code_casing():
    assert to_claude_code_name("read") == "Read"
    assert to_claude_code_name("TODOWRITE") == "TodoWrite"
    assert to_claude_code_name("my_custom_tool") == "my_custom_tool"


def test_names_map_back_to_the_callers_definition():
    tools = [Tool(name="reAd", description="d", parameters={})]
    assert from_claude_code_name("Read", tools) == "reAd"
    assert from_claude_code_name("Read", None) == "Read"
    assert from_claude_code_name("Unknown", tools) == "Unknown"


# --- id 归一化 ---


def test_tool_call_ids_are_sanitized_and_truncated():
    assert normalize_tool_call_id("call+a/b=c") == "call_a_b_c"
    assert len(normalize_tool_call_id("x" * 100)) == 64
    assert normalize_tool_call_id("plain_id-1") == "plain_id-1"


# --- 内容块 ---


def test_text_only_content_becomes_a_joined_string():
    result = convert_content_blocks([TextContent(text="a"), TextContent(text="b")])
    assert result == "a\nb"


def test_content_with_images_becomes_a_block_array():
    result = convert_content_blocks(
        [TextContent(text="look"), ImageContent(data="AAA", mime_type="image/png")]
    )
    assert result == [
        {"type": "text", "text": "look"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
    ]


def test_image_only_content_gets_a_placeholder_text_block():
    result = convert_content_blocks([ImageContent(data="AAA", mime_type="image/png")])
    assert result[0] == {"type": "text", "text": "(see attached image)"}
    assert result[1]["type"] == "image"


def test_empty_content_becomes_an_empty_string():
    assert convert_content_blocks([]) == ""


# --- user 消息 ---


def test_user_string_content_passes_through():
    assert convert_messages([user_text("hi")], False) == [{"role": "user", "content": "hi"}]


def test_blank_user_strings_are_dropped():
    assert convert_messages([user_text("   ")], False) == []


def test_blank_text_blocks_are_filtered_out():
    msg = UserMessage(content=[TextContent(text="  "), TextContent(text="keep")], timestamp=1)
    assert convert_messages([msg], False)[0]["content"] == [{"type": "text", "text": "keep"}]


def test_user_message_with_only_blank_blocks_is_dropped():
    msg = UserMessage(content=[TextContent(text="  ")], timestamp=1)
    assert convert_messages([msg], False) == []


def test_user_images_survive_the_blank_filter():
    msg = UserMessage(content=[ImageContent(data="A", mime_type="image/png")], timestamp=1)
    assert convert_messages([msg], False)[0]["content"][0]["type"] == "image"


# --- assistant 消息 ---


def test_assistant_text_and_tool_calls():
    msg = assistant([TextContent(text="sure"), ToolCall(id="c1", name="Read", arguments={"p": 1})])
    result = convert_messages([msg], False)
    assert result[0]["content"] == [
        {"type": "text", "text": "sure"},
        {"type": "tool_use", "id": "c1", "name": "Read", "input": {"p": 1}},
    ]


def test_oauth_maps_tool_names_to_claude_code_casing():
    msg = assistant([ToolCall(id="c1", name="read", arguments={})])
    assert convert_messages([msg], True)[0]["content"][0]["name"] == "Read"


def test_thinking_with_a_signature_is_replayed_as_thinking():
    msg = assistant([ThinkingContent(thinking="deep", thinking_signature="sig")])
    assert convert_messages([msg], False)[0]["content"] == [
        {"type": "thinking", "thinking": "deep", "signature": "sig"}
    ]


def test_redacted_thinking_is_replayed_as_redacted():
    msg = assistant([ThinkingContent(thinking="x", thinking_signature="blob", redacted=True)])
    assert convert_messages([msg], False)[0]["content"] == [
        {"type": "redacted_thinking", "data": "blob"}
    ]


def test_thinking_without_a_signature_degrades_to_text():
    msg = assistant([ThinkingContent(thinking="deep")])
    assert convert_messages([msg], False)[0]["content"] == [{"type": "text", "text": "deep"}]


def test_empty_signature_is_preserved_when_allowed():
    msg = assistant([ThinkingContent(thinking="deep")])
    result = convert_messages([msg], False, allow_empty_signature=True)
    assert result[0]["content"] == [{"type": "thinking", "thinking": "deep", "signature": ""}]


def test_thinking_that_is_blank_and_unsigned_is_dropped():
    msg = assistant([ThinkingContent(thinking="   "), TextContent(text="answer")])
    assert convert_messages([msg], False)[0]["content"] == [{"type": "text", "text": "answer"}]


def test_assistant_with_no_usable_blocks_is_dropped():
    assert convert_messages([assistant([TextContent(text="  ")])], False) == []


# --- 工具结果 ---


def test_consecutive_tool_results_merge_into_one_user_message():
    messages = [
        ToolResultMessage(tool_call_id="c1", tool_name="t", content=[TextContent(text="a")], timestamp=1),
        ToolResultMessage(tool_call_id="c2", tool_name="t", content=[TextContent(text="b")], timestamp=2),
    ]
    result = convert_messages(messages, False)
    assert len(result) == 1
    assert [b["tool_use_id"] for b in result[0]["content"]] == ["c1", "c2"]


def test_tool_result_carries_the_error_flag():
    msg = ToolResultMessage(
        tool_call_id="c1", tool_name="t", content=[TextContent(text="boom")], is_error=True, timestamp=1
    )
    assert convert_messages([msg], False)[0]["content"][0]["is_error"] is True


def test_tool_references_replace_the_result_content():
    msg = ToolResultMessage(
        tool_call_id="c1",
        tool_name="t",
        content=[TextContent(text="loaded ok")],
        added_tool_names=["later"],
        timestamp=1,
    )
    loaded: set[str] = set()
    tool_result, sibling = convert_tool_result(msg, False, {"later"}, loaded, lambda n: n)

    assert tool_result["content"] == [{"type": "tool_reference", "tool_name": "later"}]
    assert sibling == [{"type": "text", "text": "loaded ok"}]
    assert loaded == {"later"}


def test_a_tool_is_only_referenced_once():
    msg = ToolResultMessage(
        tool_call_id="c1", tool_name="t", added_tool_names=["later"], timestamp=1
    )
    loaded = {"later"}
    tool_result, sibling = convert_tool_result(msg, False, {"later"}, loaded, lambda n: n)
    assert tool_result["content"] == ""
    assert sibling == []


def test_non_deferred_added_tools_produce_no_reference():
    msg = ToolResultMessage(
        tool_call_id="c1", tool_name="t", added_tool_names=["other"], timestamp=1
    )
    tool_result, sibling = convert_tool_result(msg, False, {"later"}, set(), lambda n: n)
    assert tool_result["content"] == ""
    assert sibling == []


def test_displaced_content_follows_all_tool_results():
    messages = [
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="t",
            content=[TextContent(text="loaded")],
            added_tool_names=["later"],
            timestamp=1,
        ),
        ToolResultMessage(tool_call_id="c2", tool_name="t", content=[TextContent(text="plain")], timestamp=2),
    ]
    result = convert_messages(messages, False, deferred_tool_names={"later"})
    kinds = [b["type"] for b in result[0]["content"]]
    assert kinds == ["tool_result", "tool_result", "text"]


# --- 缓存标记 ---


def test_cache_control_marks_the_last_block_of_the_last_user_message():
    messages = [user_text("first"), assistant([TextContent(text="reply")]), user_text("second")]
    result = convert_messages(messages, False, cache_control=CACHE)
    assert result[-1]["content"] == [{"type": "text", "text": "second", "cache_control": CACHE}]


def test_cache_control_is_skipped_when_the_last_message_is_not_user():
    messages = [user_text("hi"), assistant([TextContent(text="reply")])]
    result = convert_messages(messages, False, cache_control=CACHE)
    assert "cache_control" not in str(result[-1])


def test_cache_control_marks_the_last_tool_result():
    messages = [ToolResultMessage(tool_call_id="c1", tool_name="t", content=[TextContent(text="ok")], timestamp=1)]
    result = convert_messages(messages, False, cache_control=CACHE)
    assert result[-1]["content"][-1]["cache_control"] == CACHE


# --- 工具序列化 ---


def make_tool(**overrides) -> Tool:
    defaults = dict(
        name="Read",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    defaults.update(overrides)
    return Tool(**defaults)


def test_tools_use_the_legacy_input_schema_by_default():
    converted = convert_tools([make_tool()], False, True, False)
    assert converted[0]["input_schema"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    assert converted[0]["eager_input_streaming"] is True
    assert "strict" not in converted[0]


def test_eager_streaming_can_be_disabled():
    converted = convert_tools([make_tool()], False, False, False)
    assert "eager_input_streaming" not in converted[0]


def test_strict_tools_send_the_full_schema():
    tool = make_tool(constrained_sampling={"type": "json_schema", "strict": "prefer"})
    converted = convert_tools([tool], False, True, True)
    assert converted[0]["strict"] is True
    assert converted[0]["input_schema"]["additionalProperties"] is False


def test_oauth_renames_tools_to_claude_code_casing():
    converted = convert_tools([make_tool(name="read")], True, True, False)
    assert converted[0]["name"] == "Read"


def test_defer_loading_is_marked():
    converted = convert_tools([make_tool()], False, True, False, None, True)
    assert converted[0]["defer_loading"] is True


def test_cache_control_only_lands_on_the_last_tool():
    converted = convert_tools([make_tool(name="a"), make_tool(name="b")], False, True, False, CACHE)
    assert "cache_control" not in converted[0]
    assert converted[1]["cache_control"] == CACHE


def test_no_tools_yields_an_empty_list():
    assert convert_tools([], False, True, False) == []
