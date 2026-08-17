from __future__ import annotations

from collections.abc import Callable

from pi_ai.types import AssistantMessage, Context, Tool, ToolResultMessage

ToolNameNormalizer = Callable[[str], str]


def split_deferred_tools(
    context: Context,
    enabled: bool,
    normalize_name: ToolNameNormalizer | None = None,
) -> tuple[list[Tool], dict[str, Tool]]:
    """把当前工具拆成"随请求发出"和"由历史加载"两组。

    延迟加载的判定：某个工具是在对话过程中通过工具结果的 added_tool_names
    引入的，且此后还没有被真正调用过。已经调用过的说明模型已经知道它了，
    仍按普通工具发出。

    返回 (immediate, deferred)，deferred 以归一化后的名字为键。
    """
    normalize = normalize_name or (lambda name: name)

    unique_tools: dict[str, Tool] = {}
    for tool in context.tools or []:
        unique_tools[normalize(tool.name)] = tool

    if not enabled:
        return list(unique_tools.values()), {}

    deferred_names: set[str] = set()
    used_names: set[str] = set()
    for message in context.messages:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if block.type == "toolCall":
                    used_names.add(normalize(block.name))
        elif isinstance(message, ToolResultMessage):
            for name in message.added_tool_names or []:
                normalized = normalize(name)
                if normalized not in used_names:
                    deferred_names.add(normalized)

    immediate: list[Tool] = []
    deferred: dict[str, Tool] = {}
    for name, tool in unique_tools.items():
        if name in deferred_names:
            deferred[name] = tool
        else:
            immediate.append(tool)
    return immediate, deferred
