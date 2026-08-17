import pytest

from pi_ai.utils.json_parse import (
    parse_json_with_repair,
    parse_streaming_json,
    repair_json,
)


def test_repair_escapes_raw_control_characters_inside_strings():
    assert repair_json('{"a":"x\ny"}') == '{"a":"x\\ny"}'


def test_repair_doubles_backslash_before_invalid_escape():
    assert repair_json('{"a":"c:\\path"}') == '{"a":"c:\\\\path"}'


def test_repair_preserves_valid_unicode_escape():
    assert repair_json('{"a":"\\u00e9"}') == '{"a":"\\u00e9"}'


def test_repair_leaves_valid_escapes_untouched():
    assert repair_json('{"a":"line\\nbreak"}') == '{"a":"line\\nbreak"}'


def test_parse_json_with_repair_recovers_from_raw_newline():
    assert parse_json_with_repair('{"a":"x\ny"}') == {"a": "x\ny"}


def test_parse_json_with_repair_reraises_when_repair_does_not_help():
    with pytest.raises(ValueError):
        parse_json_with_repair("{not json")


def test_streaming_returns_empty_dict_for_blank_input():
    assert parse_streaming_json(None) == {}
    assert parse_streaming_json("") == {}
    assert parse_streaming_json("   ") == {}


def test_streaming_parses_complete_json():
    assert parse_streaming_json('{"path":"a.txt","n":1}') == {"path": "a.txt", "n": 1}


def test_streaming_parses_truncated_object():
    assert parse_streaming_json('{"path":"a.txt","content":"hel') == {"path": "a.txt", "content": "hel"}


def test_streaming_parses_truncated_key_by_dropping_it():
    assert parse_streaming_json('{"path":"a.txt","cont') == {"path": "a.txt"}


def test_streaming_parses_truncated_nested_structures():
    assert parse_streaming_json('{"a":{"b":[1,2') == {"a": {"b": [1, 2]}}


def test_streaming_drops_trailing_comma():
    assert parse_streaming_json('{"a":1,') == {"a": 1}


def test_streaming_drops_dangling_colon():
    assert parse_streaming_json('{"a":1,"b":') == {"a": 1}


def test_streaming_returns_empty_dict_on_hopeless_input():
    assert parse_streaming_json("!!!") == {}


def test_streaming_passes_through_top_level_arrays():
    assert parse_streaming_json("[1,2,3]") == [1, 2, 3]
