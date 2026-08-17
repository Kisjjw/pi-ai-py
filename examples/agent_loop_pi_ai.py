"""和 agent_loop.py 完全一样的 agent loop，但底层换成本项目的 pi_ai。

两个文件可以对照着看，差异只有三处：
  1. 客户端：AsyncOpenAI(...) 变成 Models + Provider + Model 三件套；
  2. 历史：手搓的 items 列表变成 Context.messages，元素是消息对象而不是裸 dict；
  3. 工具：Responses 协议的平铺 schema 变成 Tool，由适配器翻译成各家协议的写法。

好处是这份 loop 不再绑死 Responses 协议。把下面的 build_* 换成别家 provider，
Anthropic / Gemini / DeepSeek 都能跑，循环体一个字不用改。

第 2 节给了三种拿到 (Models, Model) 的方式，都以 OpenAI gpt-5.5 为例，在 main()
里换一行就能切。运行：
    python examples/agent_loop_pi_ai.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from pi_ai import (
    Context,
    Model,
    Models,
    TextContent,
    Tool,
    ToolResultMessage,
    create_models,
    create_provider,
    user_text,
)
from pi_ai.api.openai_responses import openai_responses_api
from pi_ai.auth.env import env_api_key_auth
from pi_ai.catalog import load_provider_models
from pi_ai.providers import openai_provider

load_dotenv()

MAX_TURNS = 5

MY_CATALOG_DIR = Path(__file__).parent / "my_catalog"


# ---------------- 1. 工具：一个普通函数 + 一个 Tool ----------------

async def get_weather(city: str) -> str:
    return f"{city}现在晴，29 摄氏度"


async def send_message(text: str) -> str:
    print(f"    【发出去了】{text}")
    return "发送成功"


TOOL_FUNCS = {
    "get_weather": get_weather,
    "send_message": send_message,
}

# 不用再关心 strict、也不用管字段是平铺还是嵌在 function 里：
# 那些是各家协议的差异，适配器按 model.api 自己翻译。
TOOLS = [
    Tool(
        name="get_weather",
        description="查询某个城市的天气",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
    Tool(
        name="send_message",
        description="把一句话发给用户",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    ),
]


# ---------------- 2. 客户端：三种拿到 (Models, Model) 的方式 ----------------
#
# 三者的产物是一样的：一个注册好 provider 的 Models，加一个 Model。后面的 loop
# 不关心它们是怎么来的。
#
# 共同的一点：provider 的 id 不是随便起的名字，它必须等于 model.provider，
# Models 靠这个字段找到该用哪家的实现。而且适配器自己也会看这个字段——
# openai-responses 按 OPENAI_TOOL_CALL_PROVIDERS 判断要不要保留
# `callId|itemId` 复合工具调用 id，只有 id 是 "openai" 才走官方那套。
# 所以下面三种方式都用 "openai"，除了确实是另一家服务的情况。


def _require_model(models: Models, provider_id: str, model_id: str) -> Model:
    model = models.get_model(provider_id, model_id)
    if model is None:
        available = [m.id for m in models.get_models(provider_id)]
        raise LookupError(f"{provider_id} 目录里没有 {model_id!r}，现有：{available}")
    return model


def build_official() -> tuple[Models, Model]:
    """方式一：官方正版。目录、端点、鉴权全都内置，只要有 OPENAI_API_KEY。

    gpt-5.5 的上下文窗口、价格、思考级别映射、compat 开关都在
    src/pi_ai/data/openai.json 里，由上游 TypeScript 项目生成，不用你操心。
    换别家就是换这一行 provider：provider_by_id("deepseek")、anthropic_provider() 等。
    """
    models = create_models()
    models.set_provider(openai_provider())
    return models, _require_model(models, "openai", "gpt-5.5")


def build_my_catalog() -> tuple[Models, Model]:
    """方式二：自己的模型目录。内置目录里没有这个模型，或者要改价格/上限。

    examples/my_catalog/openai.json 的结构和 src/pi_ai/data/*.json 一模一样：
    最外层按 api 分组（"openai-responses"），里面是 model id -> 元数据。
    id / provider / api 这三个字段不写在 json 里，load_provider_models 会从
    文件名和分组结构补出来，所以同一份文件换个目录名就能当另一家的目录用。

    端点也来自 json 里的 baseUrl；key 用哪个环境变量与端点无关，由下面
    env_api_key_auth 的列表决定，按顺序取第一个有值的。
    """
    catalog = load_provider_models("openai", MY_CATALOG_DIR)

    models = create_models()
    models.set_provider(
        create_provider(
            id="openai",
            name="OpenAI（自带目录）",
            # 请求实际打到哪儿只看 model.base_url（或 per-call 的 base_url 选项），
            # provider 这个 base_url 适配器不读，是元数据。跟着目录填，免得两处不一致。
            base_url=next(iter(catalog.values())).base_url,
            auth={"api_key": env_api_key_auth("自带目录的 key", ["MY_OPENAI_API_KEY", "OPENAI_API_KEY"])},
            models=list(catalog.values()),
            api=openai_responses_api(),
        )
    )
    return models, _require_model(models, "openai", "gpt-5.5-little")


def build_relay() -> tuple[Models, Model]:
    """方式三：中转站。模型名和协议都跟官方一样，只有端点和 key 不同。
        id 保持 "openai"，请求体才和直连官方逐字节一致。

    只想临时打一发、不想建 provider 的话，用官方 provider 加两个 per-call 选项也行：
        models.complete_simple(model, ctx, base_url=..., api_key=...)
    显式传了 api_key 就不再查环境变量。注意 stream/complete 的每次调用都得带上。
    """
    relay_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    relayed = [
        model.model_copy(update={"base_url": relay_url})
        for model in load_provider_models("openai").values()
    ]

    models = create_models()
    models.set_provider(
        create_provider(
            id="openai",
            name="中转站",
            base_url=relay_url,
            auth={"api_key": env_api_key_auth("中转站 key", ["LLM_API_KEY"])},
            models=relayed,
            api=openai_responses_api(),
        )
    )
    return models, _require_model(models, "openai", os.getenv("LLM_MODEL") or "gpt-5.5")


# ---------------- 3. agent loop 本体 ----------------

async def run_agent(models: Models, model: Model, system: str, question: str) -> str:
    # Context 就是一次请求的全部输入。messages 会在循环里不断变长，和原来的 items 一样。
    ctx = Context(system_prompt=system, messages=[user_text(question)], tools=TOOLS)

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- 第 {turn} 轮 ---")
        print(f"  发给模型的历史: {[m.role for m in ctx.messages]}")

        reply = await models.complete_simple(model, ctx, reasoning="low")

        # 模型返回的就是一条 AssistantMessage，原样接回历史。
        # 思考块和它的签名都在 content 里，下一轮会自动回传，推理链不会断。
        ctx.messages.append(reply)
        print(f"  用量: {reply.usage.total_tokens} tokens  ${reply.usage.cost.total:.6f}")

        # 不抛异常：网络错误、4xx、超时都走这条路径。
        if reply.stop_reason == "error":
            return f"出错了: {reply.error_message}"

        calls = [block for block in reply.content if block.type == "toolCall"]

        if not calls:
            print("  模型不再调工具，结束")
            return "".join(block.text for block in reply.content if block.type == "text")

        # arguments 已经是 dict，不用自己 json.loads；配对用的是 call.id。
        for call in calls:
            print(f"  模型要调用: {call.name}({call.arguments})")
            result = await TOOL_FUNCS[call.name](**call.arguments)
            print(f"    返回: {result}")
            ctx.messages.append(
                ToolResultMessage(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    content=[TextContent(text=result)],
                )
            )

    return "达到轮数上限，强制结束"


# ---------------- 4. 跑起来 ----------------

async def main() -> None:
    # 换成 build_official() / build_my_catalog() 就切到另外两种配置方式。
    models, model = build_relay()
    print(f"用 {model.provider}/{model.id} @ {model.base_url}")

    answer = await run_agent(
        models,
        model,
        system="你是个助手。要告诉用户任何内容，都必须调用 send_message 工具。",
        question="北京天气怎么样？",
    )
    print(f"\n最终文本: {answer!r}")


if __name__ == "__main__":
    asyncio.run(main())
