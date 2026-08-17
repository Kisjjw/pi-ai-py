import pytest

from pi_ai.api.anthropic_messages_stream import AnthropicStreamParser, map_stop_reason
from pi_ai.types import Model, ModelCost, Tool


def make_model(*, model_id="claude-sonnet-4-5") -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        context_window=200000,
        max_tokens=8192,
    )


def run(parser, events):
    out = []
    for event in events:
        out.extend(parser.feed(event))
    return out


def types_of(events):
    return [e.type for e in events]


def message_start(**usage):
    base = {"input_tokens": 0, "output_tokens": 0}
    base.update(usage)
    return {"type": "message_start", "message": {"id": "msg_1", "usage": base}}


def block_start(index, content_block):
    return {"type": "content_block_start", "index": index, "content_block": content_block}


def block_delta(index, delta):
    return {"type": "content_block_delta", "index": index, "delta": delta}


def block_stop(index):
    return {"type": "content_block_stop", "index": index}


def message_delta(stop_reason="end_turn", usage=None, stop_details=None):
    event = {"type": "message_delta", "delta": {"stop_reason": stop_reason}}
    if stop_details is not None:
        event["delta"]["stop_details"] = stop_details
    if usage is not None:
        event["usage"] = usage
    return event


# --- 停止原因 ---


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "toolUse"),
        ("pause_turn", "stop"),
        ("stop_sequence", "stop"),
    ],
)
def test_stop_reason_mapping(reason, expected):
    assert map_stop_reason(reason)[0] == expected


def test_refusal_uses_the_explanation_when_present():
    stop_reason, message = map_stop_reason("refusal", {"explanation": "policy"})
    assert stop_reason == "error"
    assert message == "policy"


def test_refusal_falls_back_to_a_generic_message():
    assert map_stop_reason("refusal")[1] == "The model refused to complete the request"


def test_sensitive_maps_to_error():
    assert map_stop_reason("sensitive") == ("error", "Provider stopped with: sensitive")


def test_unknown_stop_reasons_raise():
    with pytest.raises(ValueError, match="Unhandled stop reason"):
        map_stop_reason("brand_new_reason")


# --- 用量 ---


def test_message_start_captures_initial_usage():
    parser = AnthropicStreamParser(make_model())
    run(
        parser,
        [
            message_start(
                input_tokens=100,
                output_tokens=0,
                cache_read_input_tokens=20,
                cache_creation_input_tokens=10,
                cache_creation={"ephemeral_1h_input_tokens": 4},
            )
        ],
    )
    usage = parser.output.usage
    assert usage.input == 100
    assert usage.cache_read == 20
    assert usage.cache_write == 10
    assert usage.cache_write_1h == 4
    assert usage.total_tokens == 130
    assert parser.output.response_id == "msg_1"


def test_message_delta_updates_usage_and_cost():
    parser = AnthropicStreamParser(make_model())
    run(
        parser,
        [
            message_start(input_tokens=1_000_000),
            message_delta(usage={"output_tokens": 1_000_000}),
        ],
    )
    assert parser.output.usage.output == 1_000_000
    assert parser.output.usage.cost.input == 3.0
    assert parser.output.usage.cost.output == 15.0


def test_message_delta_does_not_clobber_input_tokens_when_omitted():
    """某些代理在 message_delta 里省略 input_tokens，覆盖会把初始值冲掉。"""
    parser = AnthropicStreamParser(make_model())
    run(parser, [message_start(input_tokens=500), message_delta(usage={"output_tokens": 10})])
    assert parser.output.usage.input == 500


def test_thinking_tokens_are_recorded():
    parser = AnthropicStreamParser(make_model())
    run(
        parser,
        [
            message_start(),
            message_delta(usage={"output_tokens": 100, "output_tokens_details": {"thinking_tokens": 60}}),
        ],
    )
    assert parser.output.usage.reasoning == 60


# --- 文本 ---


def test_text_block_lifecycle():
    parser = AnthropicStreamParser(make_model())
    events = run(
        parser,
        [
            message_start(),
            block_start(0, {"type": "text", "text": ""}),
            block_delta(0, {"type": "text_delta", "text": "Hel"}),
            block_delta(0, {"type": "text_delta", "text": "lo"}),
            block_stop(0),
            message_delta(),
        ],
    )
    assert types_of(events) == ["text_start", "text_delta", "text_delta", "text_end"]
    assert parser.output.content[0].text == "Hello"
    assert parser.output.stop_reason == "stop"


def test_prefilled_text_from_block_start_is_kept():
    parser = AnthropicStreamParser(make_model())
    run(parser, [block_start(0, {"type": "text", "text": "seed"})])
    assert parser.output.content[0].text == "seed"


# --- 思考 ---


def test_thinking_block_accumulates_text_and_signature():
    parser = AnthropicStreamParser(make_model())
    events = run(
        parser,
        [
            block_start(0, {"type": "thinking", "thinking": ""}),
            block_delta(0, {"type": "thinking_delta", "thinking": "deep"}),
            block_delta(0, {"type": "signature_delta", "signature": "sig"}),
            block_stop(0),
        ],
    )
    assert types_of(events) == ["thinking_start", "thinking_delta", "thinking_end"]
    assert parser.output.content[0].thinking == "deep"
    assert parser.output.content[0].thinking_signature == "sig"


def test_signature_deltas_emit_no_events():
    parser = AnthropicStreamParser(make_model())
    events = run(
        parser,
        [
            block_start(0, {"type": "thinking", "thinking": "", "signature": "a"}),
            block_delta(0, {"type": "signature_delta", "signature": "b"}),
        ],
    )
    assert types_of(events) == ["thinking_start"]
    assert parser.output.content[0].thinking_signature == "ab"


def test_redacted_thinking_uses_a_placeholder_and_keeps_the_payload():
    parser = AnthropicStreamParser(make_model())
    run(parser, [block_start(0, {"type": "redacted_thinking", "data": "blob"})])
    block = parser.output.content[0]
    assert block.thinking == "[Reasoning redacted]"
    assert block.thinking_signature == "blob"
    assert block.redacted is True


# --- 工具调用 ---


def test_tool_call_lifecycle_parses_partial_json():
    parser = AnthropicStreamParser(make_model())
    events = run(
        parser,
        [
            block_start(0, {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}}),
            block_delta(0, {"type": "input_json_delta", "partial_json": '{"path":'}),
            block_delta(0, {"type": "input_json_delta", "partial_json": '"a.txt"}'}),
            block_stop(0),
        ],
    )
    assert types_of(events) == ["toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end"]
    call = parser.output.content[0]
    assert call.id == "tu_1"
    assert call.name == "Read"
    assert call.arguments == {"path": "a.txt"}


def test_partial_arguments_are_visible_mid_stream():
    parser = AnthropicStreamParser(make_model())
    run(
        parser,
        [
            block_start(0, {"type": "tool_use", "id": "tu_1", "name": "Write", "input": {}}),
            block_delta(0, {"type": "input_json_delta", "partial_json": '{"path":"a.txt","body":"he'}),
        ],
    )
    assert parser.output.content[0].arguments == {"path": "a.txt", "body": "he"}


def test_oauth_maps_tool_names_back_to_the_callers_definition():
    tools = [Tool(name="reAd", description="d", parameters={})]
    parser = AnthropicStreamParser(make_model(), is_oauth=True, tools=tools)
    run(parser, [block_start(0, {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}})])
    assert parser.output.content[0].name == "reAd"


def test_non_oauth_keeps_the_wire_tool_name():
    tools = [Tool(name="reAd", description="d", parameters={})]
    parser = AnthropicStreamParser(make_model(), is_oauth=False, tools=tools)
    run(parser, [block_start(0, {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}})])
    assert parser.output.content[0].name == "Read"


# --- 索引映射与其它 ---


def test_anthropic_indices_map_to_content_positions():
    """Anthropic 的 index 与 output.content 的下标不是一回事。"""
    parser = AnthropicStreamParser(make_model())
    events = run(
        parser,
        [
            block_start(5, {"type": "text", "text": ""}),
            block_start(9, {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}}),
            block_delta(9, {"type": "input_json_delta", "partial_json": "{}"}),
            block_delta(5, {"type": "text_delta", "text": "hi"}),
        ],
    )
    indices = [(e.type, e.content_index) for e in events]
    assert indices == [
        ("text_start", 0),
        ("toolcall_start", 1),
        ("toolcall_delta", 1),
        ("text_delta", 0),
    ]


def test_deltas_for_unknown_indices_are_ignored():
    parser = AnthropicStreamParser(make_model())
    assert run(parser, [block_delta(3, {"type": "text_delta", "text": "x"})]) == []


def test_unknown_event_types_are_ignored():
    parser = AnthropicStreamParser(make_model())
    assert run(parser, [{"type": "ping"}, {"type": "something_new"}]) == []


def test_unknown_content_block_types_are_ignored():
    parser = AnthropicStreamParser(make_model())
    assert run(parser, [block_start(0, {"type": "server_tool_use"})]) == []
    assert parser.output.content == []


def test_message_stop_is_tracked():
    parser = AnthropicStreamParser(make_model())
    assert parser.saw_message_stop is False
    run(parser, [{"type": "message_stop"}])
    assert parser.saw_message_stop is True


def test_finished_tool_calls_carry_no_scratch_state():
    parser = AnthropicStreamParser(make_model())
    run(
        parser,
        [
            block_start(0, {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}}),
            block_delta(0, {"type": "input_json_delta", "partial_json": "{}"}),
            block_stop(0),
        ],
    )
    dumped = parser.output.content[0].model_dump(by_alias=True, exclude_none=True)
    assert "partialJson" not in dumped
    assert "partial_json" not in dumped
