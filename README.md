# pi-ai

给 Python agent 项目省掉「请求那一层」。

35 家厂商、5 种线上协议，收敛成同一组消息类型和同一个事件流。换厂商只需要换一个 `Model` 对象，agent 循环一个字都不用改。

`@earendil-works/pi-ai` 的 Python 移植，依赖只有 `httpx` 和 `pydantic`，不装任何厂商 SDK。

## 它解决什么问题

`examples/` 下有两份**功能完全相同**的 agent loop，可以对照着看差别：

| | `agent_loop.py`（直接用厂商 SDK） | `agent_loop_pi_ai.py`（用 pi-ai） |
| --- | --- | --- |
| 对话历史 | 自己维护 `list[dict]`，逐个 `model_dump` | `Context.messages`，把返回的消息 append 回去 |
| 工具声明 | 手写协议特定的 JSON Schema，`strict` 得自己填 | `Tool(...)`，适配器翻译成各家的写法 |
| 工具参数 | `json.loads(call.arguments)` | `call.arguments` 已经是 dict |
| 思考内容 | 自己传 `include=["reasoning.encrypted_content"]` | 自动带上，签名原样回传 |
| 错误处理 | try/except 包住 | 不抛异常，看 `stop_reason` |
| 换厂商 | 重写请求与解析 | 换一个 `Model` |

## 安装

要求 Python 3.10+。

```bash
git clone git@github.com:Kisjjw/pi-ai-py.git
cd pi-ai-py
pip install -e .
```

配好厂商的 key 就能用，不需要写任何鉴权代码：

```bash
export OPENAI_API_KEY=sk-...        # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
```

### 这里有一个挺完整的示例 examples/agent_loop_pi_ai.py


## 30 秒上手

```python
import asyncio

from pi_ai import Context, create_models, user_text
from pi_ai.providers import provider_by_id


async def main():
    models = create_models()
    models.set_provider(provider_by_id("deepseek"))   # 从 DEEPSEEK_API_KEY 读 key

    model = models.get_model("deepseek", "deepseek-v4-flash")
    reply = await models.complete_simple(
        model,
        Context(system_prompt="回答简短。", messages=[user_text("用一句话解释 TCP 慢启动")]),
    )

    print(reply.content[0].text)
    print(f"{reply.usage.total_tokens} tokens, ${reply.usage.cost.total:.6f}")


asyncio.run(main())
```

模型 id 必须和目录里的一致，用 `[m.id for m in models.get_models("deepseek")]` 可以列出某家的全部模型。

## 核心概念

只有四个：

| 概念 | 作用 |
| --- | --- |
| `Model` | 一个具体模型的全部元数据：端点、上下文窗口、定价、能力开关。从内置目录取，也可以自己写 |
| `Context` | 一次请求的输入：系统提示、消息列表、工具列表 |
| `Models` | provider 集合。负责注册、鉴权、按 `model.provider` 把请求路由到对应实现 |
| `AssistantMessageEventStream` | 返回值。可以异步迭代拿增量事件，也可以 `await stream.result()` 直接拿最终消息 |

消息有三种角色，都是 pydantic 模型：

```python
user_text("你好")                                         # UserMessage 的简写
UserMessage(content=[TextContent(text="你好"), image])     # 多模态
AssistantMessage(...)                                     # 模型返回的，直接 append 回 messages
ToolResultMessage(tool_call_id=..., tool_name=..., content=[TextContent(text="结果")])
```

`AssistantMessage.content` 是个列表，元素可能是 `TextContent`、`ThinkingContent`、`ToolCall`，按模型实际产出的顺序排列。

四个入口方法：`complete_simple` / `stream_simple` 是日常用的，`complete` / `stream` 是原始接口（选项直接透传给适配器，不做上下文窗口和思考预算的换算）。

## 工具调用

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
    ctx.messages.append(reply)
    for call in (c for c in reply.content if c.type == "toolCall"):
        result = do_work(call.name, call.arguments)      # arguments 已经是 dict
        ctx.messages.append(
            ToolResultMessage(tool_call_id=call.id, tool_name=call.name, content=[TextContent(text=result)])
        )
    reply = await models.complete_simple(model, ctx)
```

参数也可以从 pydantic 模型生成：`Tool.from_pydantic(WeatherArgs, name="get_weather", description="...")`。

完整的循环见 `examples/agent_loop_pi_ai.py`。

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

## 思考 / 推理

```python
reply = await models.complete_simple(model, ctx, reasoning="high")

for block in reply.content:
    if block.type == "thinking":
        print("思考:", block.thinking)
```

级别：`off`、`minimal`、`low`、`medium`、`high`，部分模型还支持 `xhigh` / `max`。传了模型不支持的级别会自动压到最近的可用级别，不会报错。思考块的签名在多轮对话里原样回传，推理链不会断；交给另一家模型时签名会被丢弃、思考块降级成文本（签名跨厂商无效，回传会被拒）。

## 错误处理

**任何情况下都不抛异常。** 网络错误、HTTP 4xx/5xx、未配置的厂商、超时，全都变成一条 `stop_reason == "error"` 的 `AssistantMessage`：

```python
reply = await models.complete_simple(model, ctx)
if reply.stop_reason == "error":
    print("失败:", reply.error_message)
```

`stop_reason` 的取值：`stop`（正常结束）、`length`（撞到输出上限）、`toolUse`（要调工具）、`error`、`aborted`（被取消）、`deferred`。好处是流式渲染的代码不用套 try/except，错误和正常结束走同一条路径。

## 成本统计

每条回复都带算好的用量和费用：`reply.usage` 有 `input` / `output` / `cache_read` / `cache_write` / `reasoning` / `total_tokens`，`reply.usage.cost.total` 是美元金额。分层定价、Anthropic 的 1 小时缓存写入、OpenAI 的 service tier 折扣都已经算进去。`openrouter/auto` 这类动态定价模型返回负数哨兵值，表示费用要等上游结算。

## 厂商与鉴权

```python
from pi_ai.providers import all_providers, provider_by_id

models = create_models()
models.set_provider(provider_by_id("deepseek"))     # 只注册用得上的
for provider in all_providers():                     # 或者 35 家全注册
    models.set_provider(provider)
```

key 的来源按优先级：

| 方式 | 用法 |
| --- | --- |
| 单次请求指定 | `await models.complete_simple(model, ctx, api_key="sk-...")` |
| 凭据仓库 | `create_models(credentials=FileCredentialStore("auth.json"))` |
| 环境变量（默认） | `OPENAI_API_KEY`、`DEEPSEEK_API_KEY` 等，`.env` 也是走这条 |

排查配置用 `await models.get_auth("deepseek")`：返回 `None` 说明没配好，否则 `source` 字段会告诉你 key 是从哪儿读到的。

不想污染 `os.environ` 的话，用 `create_models(auth_context=DefaultAuthContext({"OPENAI_API_KEY": "sk-..."}))`。

Azure 和 Cloudflare 需要额外的环境变量（`AZURE_OPENAI_ENDPOINT`、`CLOUDFLARE_ACCOUNT_ID` 等），细节见 [docs/adding-a-provider.md](docs/adding-a-provider.md)。

<details>
<summary>支持的 35 家厂商（按协议分组）</summary>

**openai-completions**（27 家）：`ant-ling`、`baseten`、`cerebras`、`cloudflare-ai-gateway`、`cloudflare-workers-ai`、`deepseek`、`fireworks`、`github-copilot`、`groq`、`huggingface`、`moonshotai`、`moonshotai-cn`、`nvidia`、`opencode`、`opencode-go`、`openrouter`、`qwen-token-plan`、`qwen-token-plan-cn`、`qwen-token-plan-individual`、`together`、`xai`、`xiaomi`、`xiaomi-token-plan-ams`、`xiaomi-token-plan-cn`、`xiaomi-token-plan-sgp`、`zai`、`zai-coding-cn`

**anthropic-messages**（10 家）：`anthropic`、`cloudflare-ai-gateway`、`fireworks`、`github-copilot`、`kimi-coding`、`minimax`、`minimax-cn`、`opencode`、`opencode-go`、`vercel-ai-gateway`

**openai-responses**（6 家）：`openai`、`cloudflare-ai-gateway`、`github-copilot`、`opencode`、`opencode-go`、`xai`

**google-generative-ai**（2 家）：`google`、`opencode`

**azure-openai-responses**（1 家）：`azure-openai-responses`

其中 6 家同时提供多种协议，按 `model.api` 自动分派：`opencode`（4 种）、`cloudflare-ai-gateway`、`github-copilot`、`opencode-go`（各 3 种）、`fireworks`、`xai`（各 2 种）。

模型目录里还带着 4 家上游已实现、本项目尚未移植适配器的厂商（`amazon-bedrock`、`google-vertex`、`mistral`、`openai-codex`），它们的模型能被解析，但发请求会报 `unknown api`。

</details>

## 示例

| 文件 | 内容 |
| --- | --- |
| `examples/agent_loop.py` | 用厂商 SDK 手写的 agent loop，作为对照 |
| `examples/agent_loop_pi_ai.py` | 同一个 loop 换成 pi-ai，并给出三种配置方式：官方厂商、自己的模型目录、中转站 |
| `examples/my_catalog/openai.json` | 自定义模型目录的写法 |

## 接自己的服务

自建网关、中转站、私有部署，只要协议兼容就能接，用 `create_provider(id=..., base_url=..., auth=..., models=[...], api=...)` 注册一家自己的 provider 即可。两点经验：

- 中转站转发的是官方接口时，**不要手写 `Model`**。从官方目录取现成的、只换 `base_url`，价格和能力开关就都跟着官方走了。
- `Model.provider` 必须等于 `create_provider` 的 `id`：它既是路由键，也决定同协议下的厂商方言。

可运行的写法见 `examples/agent_loop_pi_ai.py` 里的三个 `build_*` 函数；完整教程（协议选择、兼容开关、自己写适配器）见 [docs/adding-a-provider.md](docs/adding-a-provider.md)。

## 开发

```bash
pytest                    # 1075 个测试，全程不打真实网络
ruff check .
mypy src

python scripts/sync_model_data.py   # 从上游 TypeScript 项目同步模型目录
```

模型目录（`src/pi_ai/data/*.json`）由上游生成，不要手改。

## 许可证

MIT
