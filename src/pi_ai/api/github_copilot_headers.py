from __future__ import annotations

from collections.abc import Sequence

from pi_ai.types import Message, ToolResultMessage, UserMessage


def infer_copilot_initiator(messages: Sequence[Message]) -> str:
    """Copilot 用 X-Initiator 区分用户发起还是 agent 发起（工具回合的后续）。"""
    if not messages:
        return "user"
    return "user" if messages[-1].role == "user" else "agent"


def has_copilot_vision_input(messages: Sequence[Message]) -> bool:
    """带图片时 Copilot 要求额外的 Copilot-Vision-Request 头。"""
    for msg in messages:
        if isinstance(msg, UserMessage) and not isinstance(msg.content, str):
            if any(block.type == "image" for block in msg.content):
                return True
        elif isinstance(msg, ToolResultMessage):
            if any(block.type == "image" for block in msg.content):
                return True
    return False


def build_copilot_dynamic_headers(messages: Sequence[Message], has_images: bool) -> dict[str, str]:
    headers = {
        "X-Initiator": infer_copilot_initiator(messages),
        "Openai-Intent": "conversation-edits",
    }
    if has_images:
        headers["Copilot-Vision-Request"] = "true"
    return headers
