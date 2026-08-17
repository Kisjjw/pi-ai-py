import base64

import pytest

from pi_ai.api.google_shared import (
    convert_messages,
    convert_tools,
    get_gemini_major_version,
    is_thinking_part,
    is_valid_thought_signature,
    map_stop_reason,
    map_tool_choice,
    requires_tool_call_id,
    resolve_google_function_calling_mode,
    resolve_thought_signature,
    retain_thought_signature,
    supports_google_strict_tool_sampling,
    supports_multimodal_function_response,
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

VALID_SIG = base64.b64encode(b"signature-bytes").decode()


def make_model(*, provider="google", model_id="gemini-2.5-flash", vision=False) -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="google-generative-ai",
        provider=provider,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        reasoning=True,
        input=["text", "image"] if vision else ["text"],
        cost=ModelCost(),
        context_window=1000000,
        max_tokens=64000,
    )


def assistant(content, *, provider="google", model_id="gemini-2.5-flash", stop_reason="stop"):
    return AssistantMessage(
        content=content,
        api="google-generative-ai",
        provider=provider,
        model=model_id,
        stop_reason=stop_reason,
        timestamp=1,
    )


# --- 思考标记与签名 ---


def test_only_thought_true_marks_thinking_content():
    """thoughtSignature 可以出现在任意 part 上，不代表该 part 是思考内容。"""
    assert is_thinking_part({"thought": True, "text": "x"}) is True
    assert is_thinking_part({"text": "x"}) is False
    assert is_thinking_part({"thoughtSignature": VALID_SIG, "text": "x"}) is False
    assert is_thinking_part({"thought": False}) is False


def test_signatures_are_retained_across_deltas():
    assert retain_thought_signature(None, "abc") == "abc"
    assert retain_thought_signature("abc", None) == "abc"
    assert retain_thought_signature("abc", "") == "abc"
    assert retain_thought_signature("abc", "def") == "def"


def test_signature_validation_requires_base64():
    assert is_valid_thought_signature(VALID_SIG) is True
    assert is_valid_thought_signature(None) is False
    assert is_valid_thought_signature("") is False
    # 长度不是 4 的倍数
    assert is_valid_thought_signature("abc") is False
    # 含非法字符
    assert is_valid_thought_signature("ab!d") is False


def test_signatures_are_dropped_across_providers():
    assert resolve_thought_signature(True, VALID_SIG) == VALID_SIG
    assert resolve_thought_signature(False, VALID_SIG) is None
    assert resolve_thought_signature(True, "not-base64!") is None


# --- 模型族判定 ---


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("gemini-2.5-flash", 2),
        ("gemini-3-pro", 3),
        ("gemini-live-2.5-flash", 2),
        ("gemma-4", None),
        ("claude-sonnet-4-5", None),
    ],
)
def test_gemini_major_version(model_id, expected):
    assert get_gemini_major_version(model_id) == expected


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("gemini-3-pro", True),
        ("gemini-2.5-flash", False),
        ("claude-sonnet-4-5", True),
        ("gpt-oss-120b", True),
        ("gemma-4", False),
    ],
)
def test_tool_call_id_requirement(model_id, expected):
    assert requires_tool_call_id(model_id) is expected


@pytest.mark.parametrize(
    "model_id,expected",
    [("gemini-3-pro", True), ("gemini-2.5-flash", False), ("claude-sonnet-4-5", True)],
)
def test_multimodal_function_response_support(model_id, expected):
    """非 Gemini 模型默认按支持处理。"""
    assert supports_multimodal_function_response(model_id) is expected


def test_strict_tool_sampling_requires_gemini_3():
    assert supports_google_strict_tool_sampling("gemini-3-pro") is True
    assert supports_google_strict_tool_sampling("gemini-2.5-flash") is False
    assert supports_google_strict_tool_sampling("claude-sonnet-4-5") is False


# --- 消息转换 ---


def test_user_string_becomes_a_text_part():
    contents = convert_messages(make_model(), Context(messages=[user_text("hi")]))
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_user_images_become_inline_data():
    msg = UserMessage(
        content=[TextContent(text="look"), ImageContent(data="AAA", mime_type="image/png")], timestamp=1
    )
    contents = convert_messages(make_model(vision=True), Context(messages=[msg]))
    assert contents[0]["parts"] == [
        {"text": "look"},
        {"inlineData": {"mimeType": "image/png", "data": "AAA"}},
    ]


def test_assistant_uses_the_model_role():
    contents = convert_messages(
        make_model(), Context(messages=[assistant([TextContent(text="hello")])])
    )
    assert contents[0] == {"role": "model", "parts": [{"text": "hello"}]}


def test_same_model_keeps_thinking_parts_with_signatures():
    msg = assistant([ThinkingContent(thinking="deep", thinking_signature=VALID_SIG)])
    contents = convert_messages(make_model(), Context(messages=[msg]))
    assert contents[0]["parts"] == [{"thought": True, "text": "deep", "thoughtSignature": VALID_SIG}]


def test_cross_model_degrades_thinking_to_plain_text():
    msg = assistant([ThinkingContent(thinking="deep", thinking_signature=VALID_SIG)], provider="other")
    contents = convert_messages(make_model(), Context(messages=[msg]))
    assert contents[0]["parts"] == [{"text": "deep"}]


def test_empty_text_with_a_signature_is_kept():
    """Gemini 会把签名挂在空文本 part 上并要求原样回传，丢了会断推理链。"""
    msg = assistant([TextContent(text="", text_signature=VALID_SIG)])
    contents = convert_messages(make_model(), Context(messages=[msg]))
    assert contents[0]["parts"] == [{"text": "", "thoughtSignature": VALID_SIG}]


def test_empty_text_without_a_signature_is_dropped():
    msg = assistant([TextContent(text="   "), TextContent(text="keep")])
    contents = convert_messages(make_model(), Context(messages=[msg]))
    assert contents[0]["parts"] == [{"text": "keep"}]


def test_empty_thinking_with_a_signature_is_kept():
    msg = assistant([ThinkingContent(thinking="", thinking_signature=VALID_SIG)])
    contents = convert_messages(make_model(), Context(messages=[msg]))
    assert contents[0]["parts"] == [{"thought": True, "text": "", "thoughtSignature": VALID_SIG}]


def test_invalid_signatures_are_stripped():
    msg = assistant([TextContent(text="hi", text_signature="not-base64!")])
    contents = convert_messages(make_model(), Context(messages=[msg]))
    assert contents[0]["parts"] == [{"text": "hi"}]


def test_tool_calls_include_ids_only_when_required():
    call = ToolCall(id="call_1", name="get_time", arguments={"tz": "UTC"})
    ctx = Context(
        messages=[
            assistant([call], stop_reason="toolUse"),
            ToolResultMessage(tool_call_id="call_1", tool_name="get_time", timestamp=2),
        ]
    )

    contents = convert_messages(make_model(model_id="gemini-2.5-flash"), ctx)
    assert contents[0]["parts"][0]["functionCall"] == {"name": "get_time", "args": {"tz": "UTC"}}

    contents = convert_messages(make_model(model_id="gemini-3-pro"), ctx)
    assert contents[0]["parts"][0]["functionCall"]["id"] == "call_1"


def test_tool_call_ids_are_sanitized_for_models_that_need_them():
    call = ToolCall(id="call+1/x", name="t", arguments={})
    ctx = Context(
        messages=[
            assistant([call], model_id="other", stop_reason="toolUse"),
            ToolResultMessage(tool_call_id="call+1/x", tool_name="t", timestamp=2),
        ]
    )
    contents = convert_messages(make_model(model_id="gemini-3-pro"), ctx)
    assert contents[0]["parts"][0]["functionCall"]["id"] == "call_1_x"


def test_assistant_with_no_parts_is_skipped():
    assert convert_messages(make_model(), Context(messages=[assistant([])])) == []


# --- 工具结果 ---


def test_tool_result_success_uses_the_output_key():
    msg = ToolResultMessage(
        tool_call_id="c1", tool_name="t", content=[TextContent(text="ok")], timestamp=1
    )
    contents = convert_messages(make_model(), Context(messages=[msg]))
    assert contents[0]["parts"][0]["functionResponse"]["response"] == {"output": "ok"}


def test_tool_result_error_uses_the_error_key():
    msg = ToolResultMessage(
        tool_call_id="c1", tool_name="t", content=[TextContent(text="boom")], is_error=True, timestamp=1
    )
    assert contents_response(msg) == {"error": "boom"}


def contents_response(msg, model=None):
    contents = convert_messages(model or make_model(), Context(messages=[msg]))
    return contents[0]["parts"][0]["functionResponse"]["response"]


def test_consecutive_tool_results_merge_into_one_user_turn():
    messages = [
        ToolResultMessage(tool_call_id="c1", tool_name="t", content=[TextContent(text="a")], timestamp=1),
        ToolResultMessage(tool_call_id="c2", tool_name="t", content=[TextContent(text="b")], timestamp=2),
    ]
    contents = convert_messages(make_model(), Context(messages=messages))
    assert len(contents) == 1
    assert len(contents[0]["parts"]) == 2


def test_gemini_3_embeds_images_in_the_function_response():
    msg = ToolResultMessage(
        tool_call_id="c1",
        tool_name="t",
        content=[TextContent(text="chart"), ImageContent(data="AAA", mime_type="image/png")],
        timestamp=1,
    )
    contents = convert_messages(make_model(model_id="gemini-3-pro", vision=True), Context(messages=[msg]))
    assert len(contents) == 1
    assert contents[0]["parts"][0]["functionResponse"]["parts"][0]["inlineData"]["data"] == "AAA"


def test_older_gemini_sends_images_in_a_separate_turn():
    msg = ToolResultMessage(
        tool_call_id="c1",
        tool_name="t",
        content=[TextContent(text="chart"), ImageContent(data="AAA", mime_type="image/png")],
        timestamp=1,
    )
    contents = convert_messages(make_model(model_id="gemini-2.5-flash", vision=True), Context(messages=[msg]))
    assert len(contents) == 2
    assert "parts" not in contents[0]["parts"][0]["functionResponse"]
    assert contents[1]["parts"][0]["text"] == "Tool result image:"


def test_image_only_tool_results_use_a_placeholder():
    msg = ToolResultMessage(
        tool_call_id="c1",
        tool_name="t",
        content=[ImageContent(data="AAA", mime_type="image/png")],
        timestamp=1,
    )
    assert contents_response(msg, make_model(model_id="gemini-3-pro", vision=True)) == {
        "output": "(see attached image)"
    }


def test_non_vision_models_drop_tool_result_images():
    msg = ToolResultMessage(
        tool_call_id="c1",
        tool_name="t",
        content=[ImageContent(data="AAA", mime_type="image/png")],
        timestamp=1,
    )
    contents = convert_messages(make_model(vision=False), Context(messages=[msg]))
    # 非视觉模型在 transform_messages 阶段就把图片换成了占位文本
    assert len(contents) == 1
    assert "parts" not in contents[0]["parts"][0]["functionResponse"]


# --- 工具转换 ---


def make_tool(**overrides) -> Tool:
    defaults = dict(
        name="get_time",
        description="d",
        parameters={"type": "object", "properties": {"tz": {"type": "string"}}, "$schema": "http://x"},
    )
    defaults.update(overrides)
    return Tool(**defaults)


def test_tools_default_to_the_json_schema_field():
    result = convert_tools([make_tool()])
    declaration = result[0]["functionDeclarations"][0]
    assert declaration["name"] == "get_time"
    assert "parametersJsonSchema" in declaration
    # 默认模式不剥元声明
    assert "$schema" in declaration["parametersJsonSchema"]


def test_openapi_mode_strips_meta_declarations():
    result = convert_tools([make_tool()], use_parameters=True)
    declaration = result[0]["functionDeclarations"][0]
    assert "$schema" not in declaration["parameters"]
    assert declaration["parameters"]["properties"] == {"tz": {"type": "string"}}


def test_meta_declarations_are_stripped_recursively():
    tool = make_tool(
        parameters={"type": "object", "properties": {"a": {"type": "string", "$comment": "x"}}}
    )
    result = convert_tools([tool], use_parameters=True)
    assert "$comment" not in result[0]["functionDeclarations"][0]["parameters"]["properties"]["a"]


def test_no_tools_yields_none():
    assert convert_tools([]) is None


# --- 工具调用模式 ---


def test_tool_choice_mapping():
    assert map_tool_choice("auto") == "AUTO"
    assert map_tool_choice("none") == "NONE"
    assert map_tool_choice("any") == "ANY"
    assert map_tool_choice("weird") == "AUTO"


def test_none_and_any_win_over_strict_mode():
    strict_tool = make_tool(constrained_sampling={"type": "json_schema", "strict": "prefer"})
    assert resolve_google_function_calling_mode([strict_tool], "none", True) == "NONE"
    assert resolve_google_function_calling_mode([strict_tool], "any", True) == "ANY"


def test_strict_tools_use_validated_mode():
    strict_tool = make_tool(constrained_sampling={"type": "json_schema", "strict": "prefer"})
    assert resolve_google_function_calling_mode([strict_tool], None, True) == "VALIDATED"
    assert resolve_google_function_calling_mode([strict_tool], "auto", True) == "VALIDATED"


def test_strict_mode_is_skipped_when_unsupported():
    strict_tool = make_tool(constrained_sampling={"type": "json_schema", "strict": "prefer"})
    assert resolve_google_function_calling_mode([strict_tool], None, False) is None


def test_no_tool_choice_and_no_strict_sends_nothing():
    assert resolve_google_function_calling_mode([make_tool()], None, True) is None


# --- 停止原因 ---


def test_stop_reason_mapping():
    assert map_stop_reason("STOP") == "stop"
    assert map_stop_reason("MAX_TOKENS") == "length"
    for reason in ("SAFETY", "RECITATION", "MALFORMED_FUNCTION_CALL", "OTHER", "BRAND_NEW"):
        assert map_stop_reason(reason) == "error"
