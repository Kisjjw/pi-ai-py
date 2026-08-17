from pi_ai.http.errors import (
    MAX_PROVIDER_ERROR_BODY_CHARS,
    ProviderHTTPError,
    format_provider_error,
    normalize_provider_error,
    truncate_error_text,
)


def test_header_lookup_is_case_insensitive():
    error = ProviderHTTPError("boom", status=429, headers={"Retry-After": "3"})
    assert error.get_header("retry-after") == "3"
    assert error.get_header("RETRY-AFTER") == "3"
    assert error.get_header("missing") is None


def test_normalize_extracts_status_and_body():
    error = ProviderHTTPError("boom", status=403, headers={}, body='{"message":"denied"}')
    norm = normalize_provider_error(error)
    assert norm.status == 403
    assert norm.body == '{"message":"denied"}'
    assert norm.message_carries_body is False


def test_normalize_marks_message_carrying_body():
    error = ProviderHTTPError('403 {"message":"denied"}', status=403, headers={}, body='{"message":"denied"}')
    norm = normalize_provider_error(error)
    assert norm.message_carries_body is True


def test_normalize_uses_parsed_error_object_when_no_raw_body():
    error = ProviderHTTPError("boom", status=400, headers={}, error={"message": "bad"})
    norm = normalize_provider_error(error)
    assert norm.body == '{"message": "bad"}'


def test_normalize_ignores_empty_error_object():
    error = ProviderHTTPError("boom", status=400, headers={}, error={})
    norm = normalize_provider_error(error)
    assert norm.body is None


def test_normalize_handles_plain_exception():
    norm = normalize_provider_error(ValueError("nope"))
    assert norm.status is None
    assert norm.body is None
    assert norm.message == "nope"
    assert norm.message_carries_body is True


def test_body_is_truncated_to_cap():
    error = ProviderHTTPError("boom", status=500, headers={}, body="x" * (MAX_PROVIDER_ERROR_BODY_CHARS + 10))
    norm = normalize_provider_error(error)
    assert norm.body is not None
    assert norm.body.endswith("... [truncated 10 chars]")


def test_format_without_prefix_surfaces_status_and_body():
    error = ProviderHTTPError("boom", status=403, headers={}, body="denied")
    assert format_provider_error(normalize_provider_error(error)) == "403: denied"


def test_format_with_prefix():
    error = ProviderHTTPError("boom", status=403, headers={}, body="denied")
    assert format_provider_error(normalize_provider_error(error), "openai") == "openai (403): denied"


def test_format_returns_message_when_body_already_included():
    error = ProviderHTTPError("403 denied", status=403, headers={}, body="denied")
    assert format_provider_error(normalize_provider_error(error)) == "403 denied"


def test_truncate_is_noop_below_cap():
    assert truncate_error_text("short", 10) == "short"
