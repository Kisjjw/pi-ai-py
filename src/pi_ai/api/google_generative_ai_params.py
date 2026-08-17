from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pi_ai.api.google_shared import (
    convert_messages,
    convert_tools,
    resolve_google_function_calling_mode,
    supports_google_strict_tool_sampling,
)
from pi_ai.types import Context, Model
from pi_ai.utils.sanitize import sanitize_surrogates

_GEMMA_4 = re.compile(r"gemma-?4")
_GEMINI_3_PRO = re.compile(r"gemini-3(?:\.\d+)?-pro")
_GEMINI_3_FLASH = re.compile(r"gemini-3(?:\.\d+)?-flash")

_FLASH_LATEST_IDS = frozenset({"gemini-flash-latest", "gemini-flash-lite-latest"})

#: SDK config 里属于 generationConfig 的字段，其余留在 REST body 顶层。
_GENERATION_CONFIG_KEYS = ("temperature", "maxOutputTokens", "thinkingConfig")


def is_gemma4_model(model: Model) -> bool:
    return bool(_GEMMA_4.search(model.id.lower()))


def is_gemini3_pro_model(model: Model) -> bool:
    return bool(_GEMINI_3_PRO.search(model.id.lower()))


def is_gemini3_flash_model(model: Model) -> bool:
    model_id = model.id.lower()
    return bool(_GEMINI_3_FLASH.search(model_id)) or model_id in _FLASH_LATEST_IDS


def get_disabled_thinking_config(model: Model) -> dict[str, Any]:
    """关闭思考时的配置。

    Google 文档：Gemini 3.1 Pro 无法关闭思考，Gemini 3 Flash / Flash-Lite 也
    不支持完全关闭。对 Gemini 3 系列改用最低的 thinkingLevel 且不带
    includeThoughts，让隐式思考对上层不可见。
    """
    if is_gemini3_pro_model(model):
        return {"thinkingLevel": "LOW"}
    if is_gemini3_flash_model(model) or is_gemma4_model(model):
        return {"thinkingLevel": "MINIMAL"}
    # Gemini 2.x 可以用 thinkingBudget = 0 真正关掉
    return {"thinkingBudget": 0}


def get_thinking_level(effort: str, model: Model) -> str:
    if is_gemini3_pro_model(model):
        return "LOW" if effort in ("minimal", "low") else "HIGH"
    if is_gemma4_model(model):
        return "MINIMAL" if effort in ("minimal", "low") else "HIGH"
    return {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}.get(effort, "HIGH")


#: 判定顺序敏感：flash-lite 必须在 flash 之前，否则会被前者吞掉。
_BUDGET_TABLES: tuple[tuple[str, dict[str, int]], ...] = (
    ("2.5-pro", {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}),
    ("2.5-flash-lite", {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}),
    ("2.5-flash", {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}),
)


def get_google_budget(
    model: Model,
    effort: str,
    custom_budgets: Mapping[str, int] | None = None,
) -> int:
    """算出思考预算。-1 表示交给模型动态决定。"""
    if custom_budgets and custom_budgets.get(effort) is not None:
        return custom_budgets[effort]

    for marker, budgets in _BUDGET_TABLES:
        if marker in model.id:
            return budgets.get(effort, -1)
    return -1


def build_params(
    model: Model,
    context: Context,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 SDK 参数形状 `{model, contents, config}`。

    这一层刻意保留 TS 侧 SDK 的形状，而不是直接产出 REST body：
    `on_payload` 拿到的必须与 TS 一致，否则上层已有的调试代码会看到不同的东西，
    M7 的金标准比对也无从对齐。REST body 由 to_rest_body 从这里转换而来。
    """
    options = options or {}
    contents = convert_messages(model, context)

    generation_config: dict[str, Any] = {}
    if options.get("temperature") is not None:
        generation_config["temperature"] = options["temperature"]
    if options.get("max_tokens") is not None:
        generation_config["maxOutputTokens"] = options["max_tokens"]

    config: dict[str, Any] = dict(generation_config)

    if context.system_prompt:
        config["systemInstruction"] = sanitize_surrogates(context.system_prompt)

    if context.tools:
        config["tools"] = convert_tools(context.tools)
        mode = resolve_google_function_calling_mode(
            context.tools, options.get("tool_choice"), supports_google_strict_tool_sampling(model.id)
        )
        if mode is not None:
            config["toolConfig"] = {"functionCallingConfig": {"mode": mode}}

    thinking = options.get("thinking")
    if thinking and thinking.get("enabled") and model.reasoning:
        thinking_config: dict[str, Any] = {"includeThoughts": True}
        if thinking.get("level") is not None:
            thinking_config["thinkingLevel"] = thinking["level"]
        elif thinking.get("budgetTokens") is not None:
            thinking_config["thinkingBudget"] = thinking["budgetTokens"]
        config["thinkingConfig"] = thinking_config
    elif model.reasoning and thinking and not thinking.get("enabled"):
        config["thinkingConfig"] = get_disabled_thinking_config(model)

    return {"model": model.id, "contents": contents, "config": config}


def to_rest_body(params: Mapping[str, Any]) -> dict[str, Any]:
    """把 SDK 参数形状转成 REST 请求体。

    差异：model 在 URL 里不在 body；config 要拆开——采样与思考参数进
    generationConfig，systemInstruction / tools / toolConfig 留在顶层，
    且 systemInstruction 必须包成 Content 对象。
    """
    config = dict(params.get("config") or {})
    body: dict[str, Any] = {"contents": params.get("contents") or []}

    generation_config = {key: config.pop(key) for key in _GENERATION_CONFIG_KEYS if key in config}
    if generation_config:
        body["generationConfig"] = generation_config

    system_instruction = config.pop("systemInstruction", None)
    if system_instruction is not None:
        body["systemInstruction"] = (
            {"parts": [{"text": system_instruction}]}
            if isinstance(system_instruction, str)
            else system_instruction
        )

    for key in ("tools", "toolConfig"):
        if key in config:
            body[key] = config.pop(key)

    # 其余键原样透传，让 sampling_params 之类的自定义字段能落到线上。
    for key, value in config.items():
        body[key] = value

    return body
