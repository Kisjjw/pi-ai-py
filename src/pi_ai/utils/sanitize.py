from __future__ import annotations

_SURROGATE_START = 0xD800
_SURROGATE_END = 0xDFFF


def sanitize_surrogates(text: str) -> str:
    """移除未配对的 Unicode 代理项。

    这类字符会让不少厂商的 JSON 序列化直接报错。

    与 TS 版的差别：TS 的字符串是 UTF-16 码元序列，需要正则区分配对与未配对；
    Python 的字符串是码点序列，合法的星文字符（emoji 等）本来就是单个码点、
    不落在 D800-DFFF 区间，所以落在该区间的必定是孤立代理项，直接滤掉即可。
    """
    if not text:
        return text
    if not any(_SURROGATE_START <= ord(char) <= _SURROGATE_END for char in text):
        return text
    return "".join(char for char in text if not _SURROGATE_START <= ord(char) <= _SURROGATE_END)
