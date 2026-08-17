from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SSEEvent:
    data: str
    event: str | None = None


@dataclass
class _Accumulator:
    event: str | None = None
    data_lines: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.event is None and not self.data_lines

    def take(self) -> SSEEvent | None:
        if self.is_empty():
            return None
        event = SSEEvent(data="\n".join(self.data_lines), event=self.event)
        self.event = None
        self.data_lines = []
        return event


def _split_lines(buffer: str) -> tuple[list[str], str]:
    r"""按 \n / \r\n / \r 切行，返回完整行与剩余不完整片段。

    末尾是单个 \r 时不切，因为后面可能跟着 \n 组成 CRLF。
    """
    lines: list[str] = []
    start = 0
    index = 0
    length = len(buffer)

    while index < length:
        char = buffer[index]
        if char == "\n":
            lines.append(buffer[start:index])
            index += 1
            start = index
            continue
        if char == "\r":
            if index == length - 1:
                break
            lines.append(buffer[start:index])
            index += 2 if buffer[index + 1] == "\n" else 1
            start = index
            continue
        index += 1

    return lines, buffer[start:]


def _apply_line(acc: _Accumulator, line: str) -> None:
    name, separator, value = line.partition(":")
    if not separator:
        name, value = line, ""
    if value.startswith(" "):
        value = value[1:]

    if name == "event":
        acc.event = value
    elif name == "data":
        acc.data_lines.append(value)


async def iter_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[SSEEvent]:
    """把字节流解析成 SSE 事件。

    支持三种换行、跨 chunk 拼接、多行 data 合并、注释行忽略。
    流结束时若还有未发出的累积内容，补发一次。
    """
    buffer = ""
    acc = _Accumulator()

    async for chunk in chunks:
        buffer += chunk.decode("utf-8", errors="replace")
        lines, buffer = _split_lines(buffer)

        for line in lines:
            if line == "":
                event = acc.take()
                if event is not None:
                    yield event
                continue
            if line.startswith(":"):
                continue
            _apply_line(acc, line)

    if buffer and not buffer.startswith(":"):
        _apply_line(acc, buffer)

    if not acc.is_empty():
        event = acc.take()
        if event is not None:
            yield event
