import json

from pi_ai.types import (
    AssistantMessage,
    Context,
    CostBreakdown,
    ImageContent,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    user_text,
)


def test_wire_field_names_match_typescript():
    block = ThinkingContent(thinking="hmm", thinking_signature="sig")
    dumped = block.model_dump(by_alias=True, exclude_none=True)
    assert dumped == {"type": "thinking", "thinking": "hmm", "thinkingSignature": "sig"}


def test_image_content_uses_mime_type_alias():
    block = ImageContent(data="aGk=", mime_type="image/png")
    assert block.model_dump(by_alias=True) == {"type": "image", "data": "aGk=", "mimeType": "image/png"}


def test_tool_result_message_wire_names():
    msg = ToolResultMessage(
        tool_call_id="call_1",
        tool_name="get_time",
        content=[TextContent(text="12:00")],
        is_error=False,
        timestamp=1,
    )
    dumped = msg.model_dump(by_alias=True, exclude_none=True)
    assert dumped["toolCallId"] == "call_1"
    assert dumped["toolName"] == "get_time"
    assert dumped["isError"] is False


def test_context_round_trips_through_json():
    ctx = Context(
        system_prompt="be brief",
        messages=[
            user_text("hi"),
            AssistantMessage(
                content=[TextContent(text="hello"), ToolCall(id="c1", name="t", arguments={"a": 1})],
                api="openai-completions",
                provider="openai",
                model="gpt-4o-mini",
                usage=Usage(cost=CostBreakdown()),
                stop_reason="toolUse",
                timestamp=2,
            ),
        ],
    )
    raw = ctx.model_dump_json(by_alias=True, exclude_none=True)
    restored = Context.model_validate_json(raw)
    assert restored.system_prompt == "be brief"
    assert restored.messages[1].content[1].name == "t"
    assert json.loads(raw)["systemPrompt"] == "be brief"


def test_user_message_content_accepts_plain_string():
    msg = user_text("hi")
    assert msg.content == "hi"
    assert msg.timestamp > 0


def test_model_compat_is_parsed_by_api():
    model = Model(
        id="m",
        name="M",
        api="openai-completions",
        provider="p",
        base_url="http://x/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
        compat={"supportsStore": False, "thinkingFormat": "deepseek"},
    )
    assert model.compat is not None
    assert model.compat.supports_store is False
    assert model.compat.thinking_format == "deepseek"
