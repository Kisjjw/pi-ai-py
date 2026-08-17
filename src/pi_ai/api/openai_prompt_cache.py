from __future__ import annotations

OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64


def clamp_openai_prompt_cache_key(key: str | None) -> str | None:
    """OpenAI 的 prompt_cache_key 上限 64 个码点，超出直接截断。"""
    if key is None:
        return None
    if len(key) <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH:
        return key
    return key[:OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH]
