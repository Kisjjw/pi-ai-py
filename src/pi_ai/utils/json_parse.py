from __future__ import annotations

import json
from typing import Any

_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')

_CONTROL_ESCAPES = {
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _is_control(char: str) -> bool:
    return 0x00 <= ord(char) <= 0x1F


def _escape_control(char: str) -> str:
    mapped = _CONTROL_ESCAPES.get(char)
    if mapped is not None:
        return mapped
    return f"\\u{ord(char):04x}"


def _is_hex4(text: str) -> bool:
    return len(text) == 4 and all(c in "0123456789abcdefABCDEF" for c in text)


def repair_json(text: str) -> str:
    """修复字符串字面量里的裸控制字符和非法转义。

    对应 TS 的 repairJson：字符串内的原始控制字符改成转义写法，
    非法转义前的反斜杠加倍。
    """
    out: list[str] = []
    in_string = False
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if not in_string:
            out.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if char == '"':
            out.append(char)
            in_string = False
            index += 1
            continue

        if char == "\\":
            next_char = text[index + 1] if index + 1 < length else None
            if next_char is None:
                out.append("\\\\")
                index += 1
                continue
            if next_char == "u" and _is_hex4(text[index + 2 : index + 6]):
                out.append("\\u" + text[index + 2 : index + 6])
                index += 6
                continue
            if next_char in _VALID_JSON_ESCAPES:
                out.append("\\" + next_char)
                index += 2
                continue
            out.append("\\\\")
            index += 1
            continue

        out.append(_escape_control(char) if _is_control(char) else char)
        index += 1

    return "".join(out)


def parse_json_with_repair(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        repaired = repair_json(text)
        if repaired != text:
            return json.loads(repaired)
        raise


def _drop_dangling_key(text: str) -> str:
    """去掉一个已读完但还没有值的 key，例如 '{"a":1,"b"' -> '{"a":1'。"""
    stripped = text.rstrip()
    if not stripped.endswith('"'):
        return stripped
    index = len(stripped) - 2
    while index >= 0:
        if stripped[index] == '"' and (index == 0 or stripped[index - 1] != "\\"):
            return stripped[:index].rstrip().rstrip(",")
        index -= 1
    return stripped


def _close_open_containers(text: str) -> str | None:
    """给未闭合的对象/数组补上闭合符。字符串未闭合或内容为空则返回 None。"""
    stack: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                stack.pop()

    if in_string:
        return None
    if not text.strip():
        return None
    return text + "".join(reversed(stack))


def _trim_dangling_separators(text: str) -> str:
    """反复去掉末尾的逗号、冒号，以及冒号前那个没有值的 key。"""
    trimmed = text.rstrip()
    while trimmed and trimmed[-1] in ",:":
        if trimmed[-1] == ":":
            trimmed = _drop_dangling_key(trimmed[:-1])
        else:
            trimmed = trimmed[:-1]
        trimmed = trimmed.rstrip()
    return trimmed


def parse_partial_json(text: str) -> Any | None:
    """解析可能被截断的 JSON。

    逐个尝试若干候选串：原串、补引号后的串、去掉悬空分隔符的串、
    以及回退到最后一个完整值处截断的串。每个候选都补齐未闭合的括号后再解析。
    """
    in_string = False
    escaped = False
    last_safe = -1

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                last_safe = index
            continue

        if char == '"':
            in_string = True
            continue
        if char in "{[":
            continue
        if char in "}]":
            last_safe = index
            continue
        if char in ",:":
            continue
        if not char.isspace():
            last_safe = index

    candidates: list[str] = []

    if not in_string:
        candidates.append(text)
    else:
        # 截断发生在字符串中间：补一个引号把它闭合
        candidates.append(text + '"')

    for candidate in list(candidates):
        trimmed = _trim_dangling_separators(candidate)
        if trimmed != candidate:
            candidates.append(trimmed)

    if last_safe >= 0:
        truncated = text[: last_safe + 1]
        candidates.append(truncated)
        candidates.append(_trim_dangling_separators(truncated))
        # 截断点落在一个 key 上时，把这个没有值的 key 也去掉
        candidates.append(_drop_dangling_key(truncated))

    for candidate in candidates:
        closed = _close_open_containers(candidate)
        if closed is None:
            continue
        try:
            return json.loads(closed)
        except ValueError:
            continue
    return None


def parse_streaming_json(text: str | None) -> Any:
    """流式场景下解析可能不完整的 JSON，永不抛异常。

    与 TS 的 parseStreamingJson 一致：解析不出来时返回空 dict。
    """
    if text is None or text.strip() == "":
        return {}

    try:
        return parse_json_with_repair(text)
    except ValueError:
        pass

    result = parse_partial_json(text)
    if result is not None:
        return result

    result = parse_partial_json(repair_json(text))
    if result is not None:
        return result

    return {}
