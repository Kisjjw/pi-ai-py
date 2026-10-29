# pi-ai

统一的异步 LLM 客户端：35 家厂商、5 种线上协议，收敛成同一组消息类型和同一个事件流。换厂商只需要换一个 `Model` 对象，agent 循环一个字都不用改。

`@earendil-works/pi-ai` 的 Python 移植，依赖只有 `httpx` 和 `pydantic`，不装任何厂商 SDK。

## 安装

要求 Python 3.10+。

```bash
pip install pi-ai-client
```

注意：**PyPI 上的包名是 `pi-ai-client`，导入名是 `pi_ai`**（`pi-ai` 这个名字过不了 PyPI 的相似名检查，注册不了）。

```python
from pi_ai import create_models
```

## 快速开始

把厂商的 key 放进环境变量，不需要写任何鉴权代码（pi-ai 只读进程环境变量，想用 `.env` 文件的话自行用 python-dotenv 之类加载）：

```bash
export DEEPSEEK_API_KEY=sk-...      # Windows PowerShell: $env:DEEPSEEK_API_KEY="sk-..."
```

```python
import asyncio

from pi_ai import Context, create_models, user_text
from pi_ai.providers import provider_by_id


async def main():
    models = create_models()
    models.set_provider(provider_by_id("deepseek"))   # 自动从 DEEPSEEK_API_KEY 读 key

    model = models.get_model("deepseek", "deepseek-flash")
    reply = await models.complete_simple(
        model,
        Context(system_prompt="回答简短。", messages=[user_text("用一句话解释 TCP 慢启动")]),
    )

    print(reply.content[0].text)
    print(f"{reply.usage.total_tokens} tokens, ${reply.usage.cost.total:.6f}")


asyncio.run(main())
```

换厂商就是换 `provider_by_id("openai")` / `provider_by_id("anthropic")` 等，再换个模型 id。OpenAI 新旗舰是 `gpt-6-astra`（同系列还有 `gpt-5.6-sol` / `gpt-5.6-terra` / `gpt-5.6-luna`），DeepSeek 现网模型是 `deepseek-flash`（V4.1 Flash，原生多模态）和 `deepseek-v4-pro`。模型 id 必须和内置目录一致，用 `[m.id for m in models.get_models("deepseek")]` 可以列出某家的全部模型。

## 三种接入方式

不管哪种方式，产物都一样：一个注册好 provider 的 `Models`，加一个 `Model`。后面的调用代码不关心它们是怎么来的。三种方式各自可运行的完整版本见 `examples/agent_loop_pi_ai.py` 里的三个 `build_*` 函数，可直接运行：`python examples/agent_loop_pi_ai.py official|catalog|relay`。

### 方式一：官方厂商（最常用）

目录、端点、鉴权全都内置，配好环境变量就能用，见上面的快速开始。

### 方式二：中转站 / 自建网关

模型和协议跟官方一样，只有端点和 key 不同。最简单的写法是在调用时直接传两个参数：

```python
reply = await models.complete_simple(
    model, ctx,
    base_url="https://my-relay.example.com/v1",
    api_key="sk-relay-key",
)
```

显式传了 `api_key` 就不再查环境变量，但每次调用都得带上。长期使用建议注册成一个 provider：

```python
import os

from pi_ai import create_models, create_provider
from pi_ai.api.openai_responses import openai_responses_api
from pi_ai.auth.env import env_api_key_auth
from pi_ai.catalog import load_provider_models

relay_url = os.environ["LLM_BASE_URL"]

# 关键：不要手写 Model。从官方目录取现成的、只换 base_url，
# 价格和能力开关就都跟着官方走了。
relayed = [
    model.model_copy(update={"base_url": relay_url})
    for model in load_provider_models("openai").values()
]

models = create_models()
models.set_provider(
    create_provider(
        id="openai",              # id 必须等于 model.provider，它是路由键
        name="中转站",
        base_url=relay_url,
        auth={"api_key": env_api_key_auth("中转站 key", ["LLM_API_KEY"])},
        models=relayed,
        api=openai_responses_api(),
    )
)
model = models.get_model("openai", "gpt-6-astra")
```

### 方式三：自定义模型目录

内置目录里没有你要的模型，或者要改价格、上下文上限时，自己写一份 json 目录（结构和 `src/pi_ai/data/*.json` 一样），用 `load_provider_models("openai", my_catalog_dir)` 加载后走方式二同样的 `create_provider` 流程。目录文件的写法见 `examples/my_catalog/openai.json`。

想接协议不兼容的私有服务、自己写适配器，见 [docs/adding-a-provider.md](docs/adding-a-provider.md)。

## 工具调用

工具用 `Tool` 声明一次，适配器翻译成各家协议的写法；返回的 `call.arguments` 已经是 dict，不用自己 `json.loads`：

```python
from pi_ai import Context, TextContent, Tool, ToolResultMessage, user_text

tools = [
    Tool(
        name="get_weather",
        description="查询某城市天气",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
]

ctx = Context(messages=[user_text("北京天气怎么样？")], tools=tools)
reply = await models.complete_simple(model, ctx)

if reply.stop_reason == "toolUse":
    ctx.messages.append(reply)          # 模型的回复原样接回历史
    for call in (c for c in reply.content if c.type == "toolCall"):
        result = do_work(call.name, call.arguments)
        ctx.messages.append(
            ToolResultMessage(tool_call_id=call.id, tool_name=call.name, content=[TextContent(text=result)])
        )
    reply = await models.complete_simple(model, ctx)
```

参数也可以从 pydantic 模型生成：`Tool.from_pydantic(WeatherArgs, name="get_weather", description="...")`。

完整的 agent loop（多轮、多工具、打印用量）见 `examples/agent_loop_pi_ai.py`；`examples/agent_loop.py` 是用厂商 SDK 手写的同款 loop，可以对照着看 pi-ai 省掉了哪些事。

## 流式输出

```python
stream = models.stream_simple(model, context)

async for event in stream:
    if event.type == "text_delta":
        print(event.delta, end="", flush=True)
    elif event.type == "thinking_delta":
        print(event.delta, end="", flush=True)
    elif event.type == "toolcall_end":
        print(f"\n调用 {event.tool_call.name}({event.tool_call.arguments})")

message = await stream.result()   # 迭代结束后拿完整消息
```

事件类型：`start`、`text_*`、`thinking_*`、`toolcall_*`（各有 `_start` / `_delta` / `_end`）、`done`、`error`。每个增量事件都带 `partial` 字段，是到此刻为止拼好的 `AssistantMessage` 快照，可以直接拿去渲染。

`stream.cancel()` 会中断底层 HTTP 请求，上游随即停止生成、也不再计费；之后 `await stream.result()` 返回 `stop_reason == "aborted"` 的消息，已收到的内容保留在 `content` 里。

## 错误处理

**任何情况下都不抛异常。** 网络错误、HTTP 4xx/5xx、未配置的厂商、超时，全都变成一条 `stop_reason == "error"` 的 `AssistantMessage`：

```python
reply = await models.complete_simple(model, ctx)
if reply.stop_reason == "error":
    print("失败:", reply.error_message)
```

`stop_reason` 的取值：`stop`（正常结束）、`length`（撞到输出上限）、`toolUse`（要调工具）、`error`、`aborted`（被取消）、`deferred`。好处是流式渲染的代码不用套 try/except，错误和正常结束走同一条路径。

## 其他能力

- **思考 / 推理**：`complete_simple(model, ctx, reasoning="high")`，级别有 `off` / `minimal` / `low` / `medium` / `high`（部分模型支持 `xhigh` / `max`），不支持的级别自动压到最近可用的。思考块在 `reply.content` 里（`block.type == "thinking"`），多轮对话里签名原样回传，推理链不会断。
- **成本统计**：每条回复自带算好的用量和费用，`reply.usage.total_tokens`、`reply.usage.cost.total`（美元）。分层定价、缓存折扣都已算进去。
- **key 的其他来源**：单次调用传 `api_key=...` 优先级最高；其次是凭据仓库 `create_models(credentials=FileCredentialStore("auth.json"))`；默认走环境变量。不想污染 `os.environ` 可用 `create_models(auth_context=DefaultAuthContext({"OPENAI_API_KEY": "sk-..."}))`。排查配置用 `await models.get_auth("deepseek")`，返回 `None` 说明没配好，否则 `source` 字段告诉你 key 从哪儿读到的。
- **原始接口**：`complete` / `stream` 把选项直接透传给适配器，不做上下文窗口和思考预算的换算，一般用 `complete_simple` / `stream_simple` 就够。

## 支持的厂商

<details>
<summary>35 家厂商（按协议分组），点开查看</summary>

**openai-completions**（27 家）：`ant-ling`、`baseten`、`cerebras`、`cloudflare-ai-gateway`、`cloudflare-workers-ai`、`deepseek`、`fireworks`、`github-copilot`、`groq`、`huggingface`、`moonshotai`、`moonshotai-cn`、`nvidia`、`opencode`、`opencode-go`、`openrouter`、`qwen-token-plan`、`qwen-token-plan-cn`、`qwen-token-plan-individual`、`together`、`xai`、`xiaomi`、`xiaomi-token-plan-ams`、`xiaomi-token-plan-cn`、`xiaomi-token-plan-sgp`、`zai`、`zai-coding-cn`

**anthropic-messages**（11 家）：`anthropic`、`cloudflare-ai-gateway`、`fireworks`、`github-copilot`、`kimi-coding`、`minimax`、`minimax-cn`、`opencode`、`opencode-go`、`openrouter`、`vercel-ai-gateway`

**openai-responses**（6 家）：`openai`、`cloudflare-ai-gateway`、`github-copilot`、`opencode`、`opencode-go`、`xai`

**google-generative-ai**（2 家）：`google`、`opencode`

**azure-openai-responses**（1 家）：`azure-openai-responses`

其中 7 家同时提供多种协议，按 `model.api` 自动分派。Azure 和 Cloudflare 需要额外的环境变量（`AZURE_OPENAI_ENDPOINT`、`CLOUDFLARE_ACCOUNT_ID` 等），细节见 [docs/adding-a-provider.md](docs/adding-a-provider.md)。

模型目录里还带着 4 家上游已实现、本项目尚未移植适配器的厂商（`amazon-bedrock`、`google-vertex`、`mistral`、`openai-codex`），它们的模型能被解析，但发请求会报 `unknown api`。

</details>

注册全部厂商：

```python
from pi_ai.providers import all_providers

models = create_models()
for provider in all_providers():
    models.set_provider(provider)
```

## 开发

```bash
git clone git@github.com:Kisjjw/pi-ai-py.git
cd pi-ai-py
pip install -e ".[dev]"

pytest                    # 1075 个测试，全程不打真实网络
ruff check .
mypy src
```

模型基础目录（`src/pi_ai/data/*.json`）由上游 TypeScript 项目生成（`python scripts/sync_model_data.py` 同步）。OpenAI、Anthropic、DeepSeek 的最新定向修订、模型 ID、价格及官方来源见 [模型目录更新记录](docs/model-catalog-updates.md)；再次同步时需保留这些修订。

---

本项目积极参与并认可 LINUX DO 社区的开源生态。
