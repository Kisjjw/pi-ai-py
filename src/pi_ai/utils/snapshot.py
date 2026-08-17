from __future__ import annotations

from pi_ai.types import AssistantContent, AssistantMessage, ToolCall


def snapshot_assistant_message(message: AssistantMessage) -> AssistantMessage:
    """冻结一份 AssistantMessage，供流式事件的 partial 使用。

    解析器为了累积增量会就地修改同一个 AssistantMessage —— 往 content 里
    append 块、给块的 text/thinking 追加字符、重新赋值 arguments、就地改 usage。
    而事件要穿过一个 asyncio 队列才到消费者手里，中间生产者还在继续修改。
    不冻结的话，消费者读到任何一个增量事件时看到的都是解析结束后的终态，
    partial 就失去了「到此刻为止」的含义。

    只需要拷到「块」这一层：文本累积在不可变的 str 上，按引用共享即可。
    需要额外处理的是两处会被就地改的可变结构：ToolCall.arguments 这个 dict，
    以及 usage 与它内嵌的 cost。
    """
    content: list[AssistantContent] = []
    for block in message.content:
        if isinstance(block, ToolCall):
            content.append(block.model_copy(update={"arguments": dict(block.arguments)}))
        else:
            content.append(block.model_copy())

    usage = message.usage.model_copy(update={"cost": message.usage.cost.model_copy()})
    return message.model_copy(update={"content": content, "usage": usage})
