import pytest

from pi_ai.api.openai_completions_stream import (
    CompletionsStreamParser,
    map_stop_reason,
    parse_chunk_usage,
)
from pi_ai.types import Model, ModelCost


def make_model(*, provider="acme", model_id="m") -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider=provider,
        base_url="https://acme.test/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=0.0),
        context_window=100000,
        max_tokens=4096,
    )


def delta_chunk(delta, **extra):
    return {"id": "resp_1", "choices": [{"index": 0, "delta": delta}], **extra}


def run(parser, chunks):
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    return events


def types_of(events):
    return [e.type for e in events]


# --- 停止原因 ---


@pytest.mark.parametrize(
    "reason,expected",
    [
        (None, "stop"),
        ("stop", "stop"),
        ("end", "stop"),
        ("length", "length"),
        ("tool_calls", "toolUse"),
        ("function_call", "toolUse"),
    ],
)
def test_stop_reason_mapping(reason, expected):
    assert map_stop_reason(reason)[0] == expected


def test_content_filter_maps_to_error_with_a_message():
    stop_reason, message = map_stop_reason("content_filter")
    assert stop_reason == "error"
    assert "content_filter" in message


def test_unknown_finish_reason_maps_to_error():
    stop_reason, message = map_stop_reason("weird")
    assert stop_reason == "error"
    assert message == "Provider finish_reason: weird"


# --- 用量 ---


def test_usage_subtracts_cache_tokens_from_input():
    usage = parse_chunk_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 300, "cache_write_tokens": 100},
        },
        make_model(),
    )
    assert usage.input == 600
    assert usage.cache_read == 300
    assert usage.cache_write == 100
    assert usage.output == 200
    assert usage.total_tokens == 1200


def test_usage_falls_back_to_prompt_cache_hit_tokens():
    usage = parse_chunk_usage(
        {"prompt_tokens": 500, "completion_tokens": 10, "prompt_cache_hit_tokens": 200}, make_model()
    )
    assert usage.cache_read == 200
    assert usage.input == 300


def test_usage_reads_reasoning_tokens():
    usage = parse_chunk_usage(
        {"prompt_tokens": 10, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 30}},
        make_model(),
    )
    assert usage.reasoning == 30
    assert usage.output == 50


def test_usage_never_goes_negative():
    usage = parse_chunk_usage(
        {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 100}}, make_model()
    )
    assert usage.input == 0


def test_usage_calculates_cost():
    usage = parse_chunk_usage({"prompt_tokens": 1_000_000, "completion_tokens": 0}, make_model())
    assert usage.cost.input == 3.0


# --- 文本 ---


def test_plain_text_stream():
    parser = CompletionsStreamParser(make_model())
    events = run(
        parser,
        [
            delta_chunk({"content": "Hel"}),
            delta_chunk({"content": "lo"}),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ],
    )
    assert types_of(events) == ["text_start", "text_delta", "text_delta", "text_end"]
    assert parser.output.content[0].text == "Hello"
    assert parser.output.stop_reason == "stop"
    assert parser.has_finish_reason is True


def test_empty_content_deltas_are_ignored():
    parser = CompletionsStreamParser(make_model())
    events = run(parser, [delta_chunk({"content": ""}), delta_chunk({"content": None})])
    assert events == []


def test_response_id_is_recorded_once():
    parser = CompletionsStreamParser(make_model())
    run(parser, [delta_chunk({"content": "a"}, id="first"), delta_chunk({"content": "b"}, id="second")])
    assert parser.output.response_id == "first"


def test_response_model_is_recorded_when_it_differs():
    parser = CompletionsStreamParser(make_model())
    run(parser, [delta_chunk({"content": "a"}, model="anthropic/claude")])
    assert parser.output.response_model == "anthropic/claude"


def test_response_model_is_not_recorded_when_it_matches():
    parser = CompletionsStreamParser(make_model())
    run(parser, [delta_chunk({"content": "a"}, model="m")])
    assert parser.output.response_model is None


# --- 思考 ---


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "reasoning_text"])
def test_each_reasoning_field_produces_thinking_events(field):
    parser = CompletionsStreamParser(make_model())
    events = run(parser, [delta_chunk({field: "think"})])
    assert types_of(events) == ["thinking_start", "thinking_delta", "thinking_end"]
    assert parser.output.content[0].thinking == "think"
    assert parser.output.content[0].thinking_signature == field


def test_first_non_empty_reasoning_field_wins():
    parser = CompletionsStreamParser(make_model())
    run(parser, [delta_chunk({"reasoning_content": "a", "reasoning": "a"})])
    assert parser.output.content[0].thinking == "a"
    assert parser.output.content[0].thinking_signature == "reasoning_content"


def test_opencode_go_renames_the_reasoning_signature():
    parser = CompletionsStreamParser(make_model(provider="opencode-go"))
    run(parser, [delta_chunk({"reasoning": "think"})])
    assert parser.output.content[0].thinking_signature == "reasoning_content"


def test_text_and_thinking_use_separate_blocks():
    parser = CompletionsStreamParser(make_model())
    events = run(parser, [delta_chunk({"content": "a", "reasoning_content": "b"})])
    assert types_of(events) == [
        "text_start",
        "text_delta",
        "thinking_start",
        "thinking_delta",
        "text_end",
        "thinking_end",
    ]


# --- 工具调用 ---


def test_tool_call_merged_by_stream_index():
    parser = CompletionsStreamParser(make_model())
    events = run(
        parser,
        [
            delta_chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_time"}}]}),
            delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"tz":'}}]}),
            delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"UTC"}'}}]}),
        ],
    )
    assert types_of(events) == [
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
    ]
    call = parser.output.content[0]
    assert call.id == "c1"
    assert call.name == "get_time"
    assert call.arguments == {"tz": "UTC"}


def test_tool_call_merged_by_id_when_index_is_absent():
    parser = CompletionsStreamParser(make_model())
    run(
        parser,
        [
            delta_chunk({"tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{"}}]}),
            delta_chunk({"tool_calls": [{"id": "c1", "function": {"arguments": "}"}}]}),
        ],
    )
    assert len(parser.output.content) == 1


def test_id_arriving_later_is_attached():
    parser = CompletionsStreamParser(make_model())
    run(
        parser,
        [
            delta_chunk({"tool_calls": [{"index": 0, "function": {"name": "t"}}]}),
            delta_chunk({"tool_calls": [{"index": 0, "id": "late", "function": {"arguments": "{}"}}]}),
        ],
    )
    assert parser.output.content[0].id == "late"


def test_two_tool_calls_produce_two_blocks():
    parser = CompletionsStreamParser(make_model())
    run(
        parser,
        [
            delta_chunk(
                {
                    "tool_calls": [
                        {"index": 0, "id": "c1", "function": {"name": "a", "arguments": "{}"}},
                        {"index": 1, "id": "c2", "function": {"name": "b", "arguments": "{}"}},
                    ]
                }
            )
        ],
    )
    assert [b.name for b in parser.output.content] == ["a", "b"]


def test_partial_arguments_are_progressively_parsed():
    parser = CompletionsStreamParser(make_model())
    truncated = '{"path":"a.txt","content":"he'
    call = {"index": 0, "id": "c1", "function": {"name": "w", "arguments": truncated}}
    events = list(parser.feed(delta_chunk({"tool_calls": [call]})))
    assert events[-1].partial.content[0].arguments == {"path": "a.txt", "content": "he"}


def test_grammar_tool_input_is_wrapped_as_json_deltas():
    parser = CompletionsStreamParser(make_model(), {"patch": "input"})
    events = run(
        parser,
        [
            delta_chunk({"tool_calls": [{"index": 0, "id": "c1", "custom": {"name": "patch", "input": "di"}}]}),
            delta_chunk({"tool_calls": [{"index": 0, "custom": {"input": "ff"}}]}),
        ],
    )
    deltas = [e.delta for e in events if e.type == "toolcall_delta"]
    assert deltas[0] == '{"input":"di'
    assert deltas[1] == "ff"
    assert deltas[2] == '"}'
    assert parser.output.content[0].arguments == {"input": "diff"}


def test_unknown_grammar_tool_falls_back_to_the_input_property():
    parser = CompletionsStreamParser(make_model())
    run(parser, [delta_chunk({"tool_calls": [{"index": 0, "id": "c1", "custom": {"name": "x", "input": "v"}}]})])
    assert parser.output.content[0].arguments == {"input": "v"}


# --- reasoning_details ---


def test_reasoning_detail_attaches_to_a_known_tool_call():
    parser = CompletionsStreamParser(make_model())
    run(
        parser,
        [
            delta_chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "t", "arguments": "{}"}}]}),
            delta_chunk({"reasoning_details": [{"type": "reasoning.encrypted", "id": "c1", "data": "blob"}]}),
        ],
    )
    assert parser.output.content[0].thought_signature == (
        '{"type":"reasoning.encrypted","id":"c1","data":"blob"}'
    )


def test_reasoning_detail_arriving_first_is_applied_later():
    parser = CompletionsStreamParser(make_model())
    run(
        parser,
        [
            delta_chunk({"reasoning_details": [{"type": "reasoning.encrypted", "id": "c1", "data": "blob"}]}),
            delta_chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "t", "arguments": "{}"}}]}),
        ],
    )
    assert parser.output.content[0].thought_signature is not None


def test_malformed_reasoning_details_are_ignored():
    parser = CompletionsStreamParser(make_model())
    run(
        parser,
        [
            delta_chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "t", "arguments": "{}"}}]}),
            delta_chunk({"reasoning_details": [{"type": "other", "id": "c1"}, {"type": "reasoning.encrypted"}]}),
        ],
    )
    assert parser.output.content[0].thought_signature is None


# --- 用量位置与收尾 ---


def test_usage_is_read_from_the_chunk():
    parser = CompletionsStreamParser(make_model())
    run(parser, [delta_chunk({"content": "a"}, usage={"prompt_tokens": 10, "completion_tokens": 5})])
    assert parser.output.usage.input == 10


def test_usage_is_read_from_the_choice_for_moonshot_style_providers():
    parser = CompletionsStreamParser(make_model())
    parser_events = list(
        parser.feed(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "a"}, "usage": {"prompt_tokens": 7, "completion_tokens": 1}}
                ]
            }
        )
    )
    assert parser_events
    assert parser.output.usage.input == 7


def test_finish_clears_streaming_scratch_state():
    parser = CompletionsStreamParser(make_model())
    run(parser, [delta_chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "t", "arguments": "{}"}}]})])
    dumped = parser.output.content[0].model_dump(by_alias=True, exclude_none=True)
    assert set(dumped) <= {"type", "id", "name", "arguments", "thoughtSignature", "namespace"}
    assert "partialArgs" not in dumped
    assert "streamIndex" not in dumped


def test_infer_missing_stop_reason_uses_tool_calls():
    parser = CompletionsStreamParser(make_model())
    run(parser, [delta_chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "t", "arguments": "{}"}}]})])
    parser.infer_missing_stop_reason()
    assert parser.output.stop_reason == "toolUse"


def test_infer_missing_stop_reason_defaults_to_stop():
    parser = CompletionsStreamParser(make_model())
    run(parser, [delta_chunk({"content": "a"})])
    parser.infer_missing_stop_reason()
    assert parser.output.stop_reason == "stop"


def test_non_dict_chunks_are_ignored():
    parser = CompletionsStreamParser(make_model())
    assert list(parser.feed("not a dict")) == []
    assert list(parser.feed(None)) == []


def test_chunks_without_choices_are_ignored():
    parser = CompletionsStreamParser(make_model())
    assert list(parser.feed({"id": "x"})) == []
