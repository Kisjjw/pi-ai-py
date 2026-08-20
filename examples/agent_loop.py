"""最小可运行的 agent loop 示例。

一个文件看完全部流程，不依赖项目里其它模块。工具就是普通的 Python 函数，
没有 MCP、没有 FastAPI、没有流式，只保留「模型 ↔ 工具」来回对话这一件事。

运行前在 .env 里配好 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，然后：
    python examples/agent_loop.py
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "")
MAX_TURNS = 5


# ---------------- 1. 工具：一个普通函数 + 一份 JSON Schema ----------------

async def get_weather(city: str) -> str:
    return f"{city}现在晴，29 摄氏度"


async def send_message(text: str) -> str:
    print(f"    【发出去了】{text}")
    return "发送成功"


# 名字 -> 真正要执行的函数
TOOL_FUNCS = {
    "get_weather": get_weather,
    "send_message": send_message,
}

# 告诉模型有哪些工具可用。Responses 协议里这些字段是平铺的，
# strict 必须显式给 false，否则默认 true 会要求所有参数都是必填。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "查询某个城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "send_message",
        "description": "把一句话发给用户",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "strict": False,
    },
]


# ---------------- 2. agent loop 本体 ----------------

async def run_agent(client: AsyncOpenAI, system: str, question: str) -> str:
    # items 就是全部对话历史。它会在循环里不断变长，这是理解整个流程的关键。
    items: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- 第 {turn} 轮 ---")
        print(f"  发给模型的历史: {[i.get('type') or i.get('role') for i in items]}")

        response = await client.responses.create(
            model=MODEL,
            input=items,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            store=False,
            include=["reasoning.encrypted_content"],
        )

        # 把模型这一轮产出的 item 原样接到历史后面。
        # 里面除了 function_call 还可能有 reasoning，一起带上模型下轮才记得自己在想什么。
        items += [item.model_dump(mode="json", exclude_none=True) for item in response.output]

        calls = [item for item in response.output if item.type == "function_call"]

        # 没有要调的工具了，说明模型说完了，退出循环
        if not calls:
            print(f"  模型不再调工具，结束")
            return response.output_text

        # 逐个执行，每个结果作为一个独立 item 追加，用 call_id 和调用配对
        for call in calls:
            args = json.loads(call.arguments or "{}")
            print(f"  模型要调用: {call.name}({args})")
            result = await TOOL_FUNCS[call.name](**args)
            print(f"    返回: {result}")
            items.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": result,
            })

    return "达到轮数上限，强制结束"


# ---------------- 3. 跑起来 ----------------

async def main() -> None:
    client = AsyncOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
    )
    answer = await run_agent(
        client,
        system="你是个助手。要告诉用户任何内容，都必须调用 send_message 工具。",
        question="北京天气怎么样？",
    )
    print(f"\n最终文本: {answer!r}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
