import json

import pytest

from pi_ai.api.openai_responses_stream import ResponsesStreamParser, map_stop_reason
from pi_ai.http.errors import ProviderHTTPError
from pi_ai.types import Model, ModelCost


def make_model(*, model_id="gpt-5") -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=0.0),
        context_window=200000,
        max_tokens=32000,
    )


def run(parser, events):
    out = []
    for event in events:
        out.extend(parser.feed(event))
    return out


def types_of(events):
    return [e.type for e in events]


def item_added(index, item):
    return {"type": "response.output_item.added", "output_index": index, "item": item}


def item_done(index, item):
    return {"type": "response.output_item.done", "output_index": index, "item": item}


def completed(**response):
    base = {"id": "resp_1", "status": "completed", "output": []}
    base.update(response)
    return {"type": "response.completed", "response": base}


# --- 停止原因 ---


@pytest.mark.parametrize(
    "status,expected",
    [("completed", "stop"), ("failed", "error"), ("cancelled", "error"), ("in_progress", "stop"), ("queued", "stop")],
)
def test_stop_reason_mapping(status, expected):
    assert map_stop_reason(status)[0] == expected


def test_incomplete_due_to_max_tokens_is_length():
    assert map_stop_reason("incomplete", "max_output_tokens") == ("length", None)


def test_other_incomplete_reasons_are_errors():
    stop_reason, message = map_stop_reason("incomplete", "content_filter")
    assert stop_reason == "error"
    assert "content_filter" in message


def test_incomplete_without_a_reason_still_errors():
    assert map_stop_reason("incomplete")[0] == "error"


def test_missing_status_defaults_to_stop():
    assert map_stop_reason(None) == ("stop", None)


def test_unknown_status_raises():
    with pytest.raises(ValueError, match="Unhandled stop reason"):
        map_stop_reason("brand_new")


# --- 文本 ---


def test_text_item_lifecycle():
    parser = ResponsesStreamParser(make_model())
    events = run(
        parser,
        [
            {"type": "response.created", "response": {"id": "resp_1"}},
            item_added(0, {"type": "message", "id": "msg_1"}),
            {"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "lo"},
            item_done(
                0,
                {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": "Hello"}],
                },
            ),
            completed(),
        ],
    )
    assert types_of(events) == ["text_start", "text_delta", "text_delta", "text_end"]
    assert parser.output.content[0].text == "Hello"
    assert parser.output.response_id == "resp_1"
    assert parser.output.stop_reason == "stop"


def test_text_signature_is_recorded_on_completion():
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            item_added(0, {"type": "message", "id": "msg_1"}),
            item_done(0, {"type": "message", "id": "msg_1", "content": [], "phase": "final_answer"}),
        ],
    )
    signature = json.loads(parser.output.content[0].text_signature)
    assert signature == {"v": 1, "id": "msg_1", "phase": "final_answer"}


def test_final_answer_phase_sets_stop_early():
    parser = ResponsesStreamParser(make_model())
    run(parser, [item_added(0, {"type": "message", "id": "m", "phase": "final_answer"})])
    assert parser.output.stop_reason == "stop"


def test_refusal_deltas_append_to_the_text_block():
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            item_added(0, {"type": "message", "id": "m"}),
            {"type": "response.refusal.delta", "output_index": 0, "delta": "I cannot"},
        ],
    )
    assert parser.output.content[0].text == "I cannot"


def test_refusal_content_is_read_on_completion():
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            item_added(0, {"type": "message", "id": "m"}),
            item_done(0, {"type": "message", "id": "m", "content": [{"type": "refusal", "refusal": "nope"}]}),
        ],
    )
    assert parser.output.content[0].text == "nope"


# --- 推理 ---


def test_reasoning_summary_deltas_accumulate():
    parser = ResponsesStreamParser(make_model())
    events = run(
        parser,
        [
            item_added(0, {"type": "reasoning", "id": "rs_1"}),
            {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "think"},
            {"type": "response.reasoning_summary_part.done", "output_index": 0},
            {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "more"},
        ],
    )
    assert types_of(events) == ["thinking_start", "thinking_delta", "thinking_delta", "thinking_delta"]
    assert parser.output.content[0].thinking == "think\n\nmore"


def test_reasoning_text_deltas_also_accumulate():
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            item_added(0, {"type": "reasoning", "id": "rs_1"}),
            {"type": "response.reasoning_text.delta", "output_index": 0, "delta": "raw"},
        ],
    )
    assert parser.output.content[0].thinking == "raw"


def test_reasoning_signature_is_the_whole_item():
    parser = ResponsesStreamParser(make_model())
    item = {"type": "reasoning", "id": "rs_1", "summary": [{"text": "final"}], "encrypted_content": "blob"}
    run(parser, [item_added(0, {"type": "reasoning", "id": "rs_1"}), item_done(0, item)])
    assert json.loads(parser.output.content[0].thinking_signature) == item
    assert parser.output.content[0].thinking == "final"


def test_reasoning_falls_back_to_content_then_accumulated_text():
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            item_added(0, {"type": "reasoning", "id": "rs_1"}),
            item_done(0, {"type": "reasoning", "id": "rs_1", "content": [{"text": "from content"}]}),
        ],
    )
    assert parser.output.content[0].thinking == "from content"

    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            item_added(0, {"type": "reasoning", "id": "rs_1"}),
            {"type": "response.reasoning_text.delta", "output_index": 0, "delta": "streamed"},
            item_done(0, {"type": "reasoning", "id": "rs_1"}),
        ],
    )
    assert parser.output.content[0].thinking == "streamed"


def test_encrypted_content_is_backfilled_from_the_terminal_response():
    """Azure 可能只在 response.completed 里给 encrypted_content。"""
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            item_added(0, {"type": "reasoning", "id": "rs_1"}),
            item_done(0, {"type": "reasoning", "id": "rs_1", "summary": [{"text": "s"}]}),
            completed(output=[{"type": "reasoning", "id": "rs_1", "encrypted_content": "late-blob"}]),
        ],
    )
    signature = json.loads(parser.output.content[0].thinking_signature)
    assert signature["encrypted_content"] == "late-blob"


def test_backfill_does_not_overwrite_an_existing_signature():
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            item_added(0, {"type": "reasoning", "id": "rs_1"}),
            item_done(0, {"type": "reasoning", "id": "rs_1", "encrypted_content": "original"}),
            completed(output=[{"type": "reasoning", "id": "rs_1", "encrypted_content": "late"}]),
        ],
    )
    assert json.loads(parser.output.content[0].thinking_signature)["encrypted_content"] == "original"


# --- 工具调用 ---


def test_function_call_lifecycle():
    parser = ResponsesStreamParser(make_model())
    events = run(
        parser,
        [
            item_added(0, {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_time"}),
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"tz":'},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '"UTC"}'},
            item_done(
                0,
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_time",
                    "arguments": '{"tz":"UTC"}',
                },
            ),
            completed(),
        ],
    )
    assert types_of(events) == ["toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end"]
    call = parser.output.content[0]
    assert call.id == "call_1|fc_1"
    assert call.name == "get_time"
    assert call.arguments == {"tz": "UTC"}
    assert parser.output.stop_reason == "toolUse"


def test_arguments_done_emits_only_the_new_suffix():
    parser = ResponsesStreamParser(make_model())
    events = run(
        parser,
        [
            item_added(0, {"type": "function_call", "id": "fc_1", "call_id": "c", "name": "t"}),
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"a":'},
            {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": '{"a":1}'},
        ],
    )
    assert [e.delta for e in events if e.type == "toolcall_delta"] == ['{"a":', "1}"]


def test_arguments_done_with_a_divergent_value_emits_no_delta():
    parser = ResponsesStreamParser(make_model())
    events = run(
        parser,
        [
            item_added(0, {"type": "function_call", "id": "fc_1", "call_id": "c", "name": "t"}),
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "garbage"},
            {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": '{"a":1}'},
        ],
    )
    assert [e.delta for e in events if e.type == "toolcall_delta"] == ["garbage"]
    assert parser.output.content[0].arguments == {"a": 1}


def test_custom_tool_call_input_is_wrapped_as_json():
    parser = ResponsesStreamParser(make_model(), grammar_tool_input_properties={"patch": "input"})
    events = run(
        parser,
        [
            item_added(0, {"type": "custom_tool_call", "id": "ctc_1", "call_id": "c", "name": "patch"}),
            {"type": "response.custom_tool_call_input.delta", "output_index": 0, "delta": "di"},
            {"type": "response.custom_tool_call_input.done", "output_index": 0, "input": "diff"},
        ],
    )
    deltas = [e.delta for e in events if e.type == "toolcall_delta"]
    assert deltas[0] == '{"input":"di'
    assert deltas[1] == 'ff"}'
    assert parser.output.content[0].arguments == {"input": "diff"}


def test_namespace_from_the_done_item_is_applied():
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            item_added(0, {"type": "function_call", "id": "fc_1", "call_id": "c", "name": "t"}),
            item_done(
                0,
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "c",
                    "name": "t",
                    "arguments": "{}",
                    "namespace": "ns",
                },
            ),
        ],
    )
    assert parser.output.content[0].namespace == "ns"


def test_items_done_without_a_prior_added_still_work():
    parser = ResponsesStreamParser(make_model())
    events = run(
        parser,
        [item_done(0, {"type": "message", "id": "m", "content": [{"type": "output_text", "text": "hi"}]})],
    )
    assert types_of(events) == ["text_start", "text_end"]
    assert parser.output.content[0].text == "hi"


def test_finished_tool_calls_carry_no_scratch_state():
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            item_added(0, {"type": "function_call", "id": "fc_1", "call_id": "c", "name": "t"}),
            item_done(0, {"type": "function_call", "id": "fc_1", "call_id": "c", "name": "t", "arguments": "{}"}),
        ],
    )
    dumped = parser.output.content[0].model_dump(by_alias=True, exclude_none=True)
    assert "partialJson" not in dumped
    assert "customInput" not in dumped


# --- 用量与终态 ---


def test_usage_subtracts_cache_tokens_from_input():
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            completed(
                usage={
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "total_tokens": 1200,
                    "input_tokens_details": {"cached_tokens": 300, "cache_write_tokens": 100},
                    "output_tokens_details": {"reasoning_tokens": 50},
                }
            )
        ],
    )
    usage = parser.output.usage
    assert usage.input == 600
    assert usage.cache_read == 300
    assert usage.cache_write == 100
    assert usage.reasoning == 50
    assert usage.total_tokens == 1200
    assert usage.cost.total > 0


def test_incomplete_response_maps_to_length():
    parser = ResponsesStreamParser(make_model())
    run(
        parser,
        [
            {
                "type": "response.incomplete",
                "response": {
                    "id": "r",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            }
        ],
    )
    assert parser.output.stop_reason == "length"
    assert parser.output.raw_stop_reason == "incomplete.max_output_tokens"


def test_service_tier_pricing_hook_is_applied():
    applied = []

    parser = ResponsesStreamParser(
        make_model(),
        service_tier="flex",
        apply_service_tier_pricing=lambda usage, tier: applied.append(tier),
    )
    run(parser, [completed(usage={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11})])
    assert applied == ["flex"]


def test_response_service_tier_wins_over_the_requested_one():
    applied = []
    parser = ResponsesStreamParser(
        make_model(),
        service_tier="flex",
        apply_service_tier_pricing=lambda usage, tier: applied.append(tier),
    )
    run(parser, [completed(service_tier="priority", usage={"input_tokens": 1})])
    assert applied == ["priority"]


def test_terminal_event_is_tracked():
    parser = ResponsesStreamParser(make_model())
    assert parser.saw_terminal_event is False
    run(parser, [completed()])
    assert parser.saw_terminal_event is True


def test_error_events_raise():
    parser = ResponsesStreamParser(make_model())
    with pytest.raises(ProviderHTTPError, match="Error Code 500"):
        run(parser, [{"type": "error", "code": 500, "message": "boom"}])


def test_response_failed_raises_with_the_error_details():
    parser = ResponsesStreamParser(make_model())
    with pytest.raises(ProviderHTTPError, match="rate_limit: too fast"):
        run(
            parser,
            [
                {
                    "type": "response.failed",
                    "response": {"status": "failed", "error": {"code": "rate_limit", "message": "too fast"}},
                }
            ],
        )


def test_response_failed_falls_back_to_incomplete_details():
    parser = ResponsesStreamParser(make_model())
    with pytest.raises(ProviderHTTPError, match="incomplete: content_filter"):
        run(
            parser,
            [
                {
                    "type": "response.failed",
                    "response": {"status": "failed", "incomplete_details": {"reason": "content_filter"}},
                }
            ],
        )


def test_unknown_events_are_ignored():
    parser = ResponsesStreamParser(make_model())
    assert run(parser, [{"type": "response.in_progress"}, {"type": "something.new"}]) == []
