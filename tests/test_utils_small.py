from pi_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    user_text,
)
from pi_ai.utils.estimate import (
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_text_and_image_content_tokens,
    estimate_text_tokens,
)
from pi_ai.utils.hash import short_hash
from pi_ai.utils.provider_env import get_provider_env_value
from pi_ai.utils.sanitize import sanitize_surrogates

# 基准值由 TS 侧 src/utils/hash.ts 的 shortHash 实测得出。
TS_HASH_BASELINE = {
    "": "k4n83c7h0j2b",
    "hello": "1h6qa0qrowduu",
    "call_abc|item_xyz": "5r62khfrcycv",
    "call_9BdBqTMSDgH0mFvBtRHTNPUE|rs_68f0b4a8c0f88191b2c3d4e5f6a7b8c9": "rgzox1tb0glv",
    "中文测试": "1o8yehk11ajja5",
    "a" * 500: "d33jejlgylnv",
}


def test_short_hash_matches_typescript_output():
    for text, expected in TS_HASH_BASELINE.items():
        assert short_hash(text) == expected, f"mismatch for {text[:32]!r}"


def test_short_hash_uses_base36_alphabet():
    value = short_hash("call_abc|item_xyz")
    assert value
    assert all(c in "0123456789abcdefghijklmnopqrstuvwxyz" for c in value)


def test_sanitize_preserves_paired_astral_characters():
    assert sanitize_surrogates("Hello 🙈 World") == "Hello 🙈 World"
    assert sanitize_surrogates("日本語テキスト") == "日本語テキスト"


def test_sanitize_removes_lone_high_surrogate():
    assert sanitize_surrogates("Text \ud83d here") == "Text  here"


def test_sanitize_removes_lone_low_surrogate():
    assert sanitize_surrogates("Text \udc4d here") == "Text  here"


def test_sanitize_is_noop_for_clean_text():
    assert sanitize_surrogates("plain") == "plain"
    assert sanitize_surrogates("") == ""


def test_estimate_text_tokens_rounds_up():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abc") == 1
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2


def test_estimate_counts_images_as_fixed_char_budget():
    assert estimate_text_and_image_content_tokens([ImageContent(data="x", mime_type="image/png")]) == 1200


def test_estimate_accepts_plain_string_content():
    assert estimate_text_and_image_content_tokens("abcd") == 1


def test_estimate_message_tokens_covers_every_block_type():
    assistant = AssistantMessage(
        content=[
            TextContent(text="a" * 4),
            ThinkingContent(thinking="b" * 4),
            ToolCall(id="1", name="ab", arguments={}),
        ],
        api="openai-completions",
        provider="p",
        model="m",
        stop_reason="stop",
    )
    # 4 + 4 + len("ab") + len("{}") = 12 chars -> 3 tokens
    assert estimate_message_tokens(assistant) == 3


def test_calculate_context_tokens_prefers_total_then_sums_parts():
    assert calculate_context_tokens(Usage(total_tokens=42)) == 42
    assert calculate_context_tokens(Usage(input=1, output=2, cache_read=3, cache_write=4)) == 10


def test_estimate_uses_last_assistant_usage_plus_trailing_messages():
    ctx = Context(
        messages=[
            UserMessage(content="x" * 400, timestamp=1),
            AssistantMessage(
                content=[TextContent(text="done")],
                api="openai-completions",
                provider="p",
                model="m",
                usage=Usage(total_tokens=1000),
                stop_reason="stop",
                timestamp=2,
            ),
            UserMessage(content="y" * 8, timestamp=3),
        ]
    )
    estimate = estimate_context_tokens(ctx)
    assert estimate.usage_tokens == 1000
    assert estimate.trailing_tokens == 2
    assert estimate.tokens == 1002
    assert estimate.last_usage_index == 1


def test_estimate_ignores_usage_from_aborted_assistant():
    ctx = Context(
        messages=[
            AssistantMessage(
                content=[TextContent(text="partial")],
                api="openai-completions",
                provider="p",
                model="m",
                usage=Usage(total_tokens=1000),
                stop_reason="aborted",
                timestamp=1,
            ),
        ]
    )
    estimate = estimate_context_tokens(ctx)
    assert estimate.last_usage_index is None
    assert estimate.usage_tokens == 0


def test_estimate_ignores_usage_superseded_by_a_newer_message():
    # timestamp=200 的消息插在前面，之后出现的 timestamp=150 assistant
    # 其用量不再描述当前前缀，应被忽略。
    ctx = Context(
        messages=[
            UserMessage(content="a" * 8, timestamp=200),
            AssistantMessage(
                content=[TextContent(text="stale")],
                api="openai-completions",
                provider="p",
                model="m",
                usage=Usage(total_tokens=9999),
                stop_reason="stop",
                timestamp=150,
            ),
        ]
    )
    estimate = estimate_context_tokens(ctx)
    assert estimate.last_usage_index is None


def test_estimate_adds_system_prompt_and_tools_when_no_usage_exists():
    tool = Tool(name="t", description="d", parameters={"type": "object"})
    ctx = Context(system_prompt="s" * 8, messages=[user_text("x" * 4)], tools=[tool])
    estimate = estimate_context_tokens(ctx)
    without_prefix = estimate_message_tokens(ctx.messages[0])
    assert estimate.tokens > without_prefix
    assert estimate.last_usage_index is None


def test_estimate_adds_only_newly_loaded_tools_after_usage():
    loaded = Tool(name="loaded", description="d" * 100, parameters={"type": "object"})
    other = Tool(name="other", description="d" * 100, parameters={"type": "object"})
    ctx = Context(
        messages=[
            AssistantMessage(
                content=[ToolCall(id="1", name="loaded", arguments={})],
                api="openai-completions",
                provider="p",
                model="m",
                usage=Usage(total_tokens=500),
                stop_reason="toolUse",
                timestamp=1,
            ),
            ToolResultMessage(
                tool_call_id="1",
                tool_name="loaded",
                content=[TextContent(text="ok")],
                added_tool_names=["loaded"],
                timestamp=2,
            ),
        ],
        tools=[loaded, other],
    )
    estimate = estimate_context_tokens(ctx)
    assert estimate.usage_tokens == 500
    # 只有 added_tool_names 里的工具计入，另一个不计
    assert estimate.trailing_tokens > 0
    assert estimate.tokens == 500 + estimate.trailing_tokens


def test_estimate_accepts_a_bare_message_list():
    estimate = estimate_context_tokens([user_text("abcd")])
    assert estimate.tokens == 1


def test_provider_env_prefers_scoped_over_process(monkeypatch):
    monkeypatch.setenv("PI_AI_ENV_TEST", "proc")
    assert get_provider_env_value("PI_AI_ENV_TEST", {"PI_AI_ENV_TEST": "scoped"}) == "scoped"
    assert get_provider_env_value("PI_AI_ENV_TEST", None) == "proc"
    assert get_provider_env_value("PI_AI_ENV_TEST", {}) == "proc"


def test_provider_env_treats_empty_string_as_absent(monkeypatch):
    monkeypatch.setenv("PI_AI_ENV_EMPTY", "")
    assert get_provider_env_value("PI_AI_ENV_EMPTY", None) is None
    assert get_provider_env_value("PI_AI_ENV_MISSING", None) is None
