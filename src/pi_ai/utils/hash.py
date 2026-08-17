from __future__ import annotations

_MASK32 = 0xFFFFFFFF
_BASE36_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def _imul(a: int, b: int) -> int:
    """等价于 JS 的 Math.imul：按 32 位有符号整数相乘。

    Python 整数是任意精度的，不截断的话结果会与 TS 分道扬镳。
    """
    result = (a * b) & _MASK32
    return result - 0x100000000 if result >= 0x80000000 else result


def _to_int32(value: int) -> int:
    value &= _MASK32
    return value - 0x100000000 if value >= 0x80000000 else value


def _to_base36(value: int) -> str:
    if value == 0:
        return "0"
    digits: list[str] = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(_BASE36_DIGITS[remainder])
    return "".join(reversed(digits))


def short_hash(text: str) -> str:
    """快速确定性哈希，用于缩短超长字符串。

    移植自 TS 的 src/utils/hash.ts。输出必须与 TS 逐字一致：这个哈希用于
    截断超长工具调用 id，两端不一致会让跨端回放的会话对不上 id。

    注意逐字符取的是 UTF-16 码元（charCodeAt），因此基本多文种平面之外的
    字符要按代理对拆开处理。
    """
    h1 = _to_int32(0xDEADBEEF)
    h2 = _to_int32(0x41C6CE57)

    for code_unit in _utf16_code_units(text):
        h1 = _imul(h1 ^ code_unit, 2654435761)
        h2 = _imul(h2 ^ code_unit, 1597334677)

    h1 = _to_int32(
        _imul(h1 ^ ((h1 & _MASK32) >> 16), 2246822507) ^ _imul(h2 ^ ((h2 & _MASK32) >> 13), 3266489909)
    )
    h2 = _to_int32(
        _imul(h2 ^ ((h2 & _MASK32) >> 16), 2246822507) ^ _imul(h1 ^ ((h1 & _MASK32) >> 13), 3266489909)
    )

    return _to_base36(h2 & _MASK32) + _to_base36(h1 & _MASK32)


def _utf16_code_units(text: str) -> list[int]:
    """把字符串拆成 UTF-16 码元序列，对齐 JS 的 charCodeAt 语义。"""
    units: list[int] = []
    for char in text:
        code_point = ord(char)
        if code_point > 0xFFFF:
            offset = code_point - 0x10000
            units.append(0xD800 + (offset >> 10))
            units.append(0xDC00 + (offset & 0x3FF))
        else:
            units.append(code_point)
    return units
