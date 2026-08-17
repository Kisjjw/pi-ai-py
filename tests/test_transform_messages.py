from pi_ai.api.transform_messages import transform_messages
from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    user_text,
)

NON_VISION_USER_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_PLACEHOLDER = "(tool image omitted: model does not support images)"


def make_model(*, provider="openai", api="openai-completions", model_id="m", vision=False) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api=api,
        provider=provider,
        base_url="http://x/v1",
        reasoning=False,
        input=["text", "image"] if vision else ["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
    )


def make_assistant(content, *, provider="openai", api="openai-completions", model_id="m", stop_reason="stop"):
    return AssistantMessage(
        content=content,
        api=api,
        provider=provider,
        model=model_id,
        stop_reason=stop_reason,
        timestamp=1,
    )


def test_user_string_content_passes_through():
    messages = transform_messages([user_text("hi")], make_model())
    assert messages[0].content == "hi"


def test_non_vision_model_replaces_user_images_with_placeholder():
    msg = UserMessage(
        content=[TextContent(text="look"), ImageContent(data="x", mime_type="image/png")],
        timestamp=1,
    )
    result = transform_messages([msg], make_model(vision=False))
    assert [b.text for b in result[0].content] == ["look", NON_VISION_USER_PLACEHOLDER]


def test_consecutive_images_collapse_to_one_placeholder():
    msg = UserMessage(
        content=[
            ImageContent(data="a", mime_type="image/png"),
            ImageContent(data="b", mime_type="image/png"),
            TextContent(text="after"),
        ],
        timestamp=1,
    )
    result = transform_messages([msg], make_model(vision=False))
    assert [b.text for b in result[0].content] == [NON_VISION_USER_PLACEHOLDER, "after"]


def test_tool_result_images_use_their_own_placeholder():
    msg = ToolResultMessage(
        tool_call_id="c1",
        tool_name="t",
        content=[ImageContent(data="a", mime_type="image/png")],
        timestamp=1,
    )
    result = transform_messages([msg], make_model(vision=False))
    assert [b.text for b in result[0].content] == [NON_VISION_TOOL_PLACEHOLDER]


def test_vision_model_keeps_images():
    msg = UserMessage(content=[ImageContent(data="a", mime_type="image/png")], timestamp=1)
    result = transform_messages([msg], make_model(vision=True))
    assert result[0].content[0].type == "image"


def test_same_model_keeps_thinking_blocks():
    msg = make_assistant([ThinkingContent(thinking="deep", thinking_signature="sig")])
    result = transform_messages([msg], make_model())
    assert result[0].content[0].type == "thinking"


def test_cross_model_converts_thinking_to_text():
    msg = make_assistant([ThinkingContent(thinking="deep")], provider="other")
    result = transform_messages([msg], make_model())
    assert result[0].content[0].type == "text"
    assert result[0].content[0].text == "deep"


def test_cross_model_drops_redacted_thinking():
    msg = make_assistant(
        [ThinkingContent(thinking="x", redacted=True, thinking_signature="s"), TextContent(text="keep")],
        provider="other",
    )
    result = transform_messages([msg], make_model())
    assert [b.type for b in result[0].content] == ["text"]


def test_same_model_keeps_redacted_thinking():
    msg = make_assistant([ThinkingContent(thinking="x", redacted=True, thinking_signature="s")])
    result = transform_messages([msg], make_model())
    assert result[0].content[0].redacted is True


def test_same_model_keeps_empty_thinking_when_it_has_a_signature():
    msg = make_assistant([ThinkingContent(thinking="", thinking_signature="sig"), TextContent(text="t")])
    result = transform_messages([msg], make_model())
    assert [b.type for b in result[0].content] == ["thinking", "text"]


def test_empty_thinking_without_signature_is_dropped():
    msg = make_assistant([ThinkingContent(thinking="   "), TextContent(text="t")])
    result = transform_messages([msg], make_model())
    assert [b.type for b in result[0].content] == ["text"]


def test_cross_model_strips_thought_signature_from_tool_calls():
    msg = make_assistant(
        [ToolCall(id="c1", name="t", arguments={}, thought_signature="sig")],
        provider="other",
    )
    result = transform_messages(
        [msg, ToolResultMessage(tool_call_id="c1", tool_name="t", timestamp=2)],
        make_model(),
    )
    assert result[0].content[0].thought_signature is None


def test_same_model_keeps_thought_signature():
    msg = make_assistant([ToolCall(id="c1", name="t", arguments={}, thought_signature="sig")])
    result = transform_messages(
        [msg, ToolResultMessage(tool_call_id="c1", tool_name="t", timestamp=2)],
        make_model(),
    )
    assert result[0].content[0].thought_signature == "sig"


def test_tool_call_ids_are_normalized_and_results_follow():
    msg = make_assistant([ToolCall(id="LONG", name="t", arguments={})], provider="other")
    result = transform_messages(
        [msg, ToolResultMessage(tool_call_id="LONG", tool_name="t", timestamp=2)],
        make_model(),
        lambda call_id, model, source: call_id.lower(),
    )
    assert result[0].content[0].id == "long"
    assert result[1].tool_call_id == "long"


def test_tool_call_ids_are_not_normalized_for_the_same_model():
    msg = make_assistant([ToolCall(id="LONG", name="t", arguments={})])
    result = transform_messages(
        [msg, ToolResultMessage(tool_call_id="LONG", tool_name="t", timestamp=2)],
        make_model(),
        lambda call_id, model, source: call_id.lower(),
    )
    assert result[0].content[0].id == "LONG"


def test_errored_assistant_messages_are_skipped():
    messages = [
        user_text("a"),
        make_assistant([TextContent(text="partial")], stop_reason="error"),
        user_text("b"),
    ]
    result = transform_messages(messages, make_model())
    assert [m.role for m in result] == ["user", "user"]


def test_aborted_assistant_messages_are_skipped():
    messages = [make_assistant([TextContent(text="partial")], stop_reason="aborted")]
    assert transform_messages(messages, make_model()) == []


def test_orphaned_tool_call_gets_a_synthetic_result_at_the_end():
    messages = [make_assistant([ToolCall(id="c1", name="t", arguments={})], stop_reason="toolUse")]
    result = transform_messages(messages, make_model())
    assert len(result) == 2
    assert result[1].role == "toolResult"
    assert result[1].tool_call_id == "c1"
    assert result[1].is_error is True
    assert result[1].content[0].text == "No result provided"


def test_orphaned_tool_call_gets_a_synthetic_result_before_a_user_message():
    messages = [
        make_assistant([ToolCall(id="c1", name="t", arguments={})], stop_reason="toolUse"),
        user_text("next"),
    ]
    result = transform_messages(messages, make_model())
    assert [m.role for m in result] == ["assistant", "toolResult", "user"]


def test_orphaned_tool_call_gets_a_synthetic_result_before_the_next_assistant():
    messages = [
        make_assistant([ToolCall(id="c1", name="t", arguments={})], stop_reason="toolUse"),
        make_assistant([TextContent(text="next")]),
    ]
    result = transform_messages(messages, make_model())
    assert [m.role for m in result] == ["assistant", "toolResult", "assistant"]


def test_existing_tool_results_are_not_duplicated():
    messages = [
        make_assistant([ToolCall(id="c1", name="t", arguments={})], stop_reason="toolUse"),
        ToolResultMessage(tool_call_id="c1", tool_name="t", content=[TextContent(text="ok")], timestamp=2),
    ]
    result = transform_messages(messages, make_model())
    assert len(result) == 2
    assert result[1].content[0].text == "ok"


def test_only_missing_results_are_synthesized():
    messages = [
        make_assistant(
            [ToolCall(id="c1", name="t", arguments={}), ToolCall(id="c2", name="t", arguments={})],
            stop_reason="toolUse",
        ),
        ToolResultMessage(tool_call_id="c1", tool_name="t", content=[TextContent(text="ok")], timestamp=2),
    ]
    result = transform_messages(messages, make_model())
    assert [m.role for m in result] == ["assistant", "toolResult", "toolResult"]
    assert result[2].tool_call_id == "c2"


def test_input_messages_are_not_mutated():
    original = make_assistant([ThinkingContent(thinking="deep")], provider="other")
    transform_messages([original], make_model())
    assert original.content[0].type == "thinking"


def test_none_content_is_normalized_to_empty_list():
    msg = make_assistant([])
    msg.content = None  # 模拟手写历史或旧会话文件
    result = transform_messages([msg], make_model())
    assert result == [] or result[0].content == []
