# 接入一家新厂商

假设现在冒出一家新厂商「Starlight」，要把它接进来。这篇按步骤走一遍，代码都能直接抄。

先明确两个概念，后面一直用：

- **`api`（协议）**：怎么说话。URL 路径长什么样、请求体字段怎么起名、SSE 事件怎么解析。
- **`id`（厂商 id）**：跟谁说话。决定路由、鉴权、以及同一协议下的厂商方言。

两者是多对多：一家厂商可以支持多种协议（`github-copilot` 有三种），一种协议也被几十家共用（`openai-completions` 有 27 家）。所以下面每一步都要把这两样分别交代清楚。

---

## 第 0 步：分诊，你要走哪条路

工作量差一个数量级，先判断新厂商的线上协议长什么样：

| 特征 | 用现成的 api |
| --- | --- |
| `POST /chat/completions`，`messages: [{role, content}]`，`tools: [{type:"function", function:{...}}]` | `openai-completions` |
| `POST /v1/messages`，`system` 独立字段，`content` 是块数组，`max_tokens` 必填 | `anthropic-messages` |
| `POST /responses`，`input` 是 item 列表，工具 schema 平铺 | `openai-responses` |
| `POST /models/<id>:streamGenerateContent`，`contents` / `parts` | `google-generative-ai` |
| 端点由资源名和部署名拼出来 | `azure-openai-responses` |
| 以上都不像 | 走**路线 B**，自己写适配器 |

绝大多数新厂商（包括所有中转站、绝大部分国产模型）都兼容前两种，走**路线 A** 就够了，几十行搞定。

判断的办法很朴素：拿厂商文档里的 curl 示例，和 `src/pi_ai/api/*_params.py` 里 `build_params` 生成的形状比一比。想直接看生成结果，运行：

```python
from pi_ai.api.openai_completions_params import build_params
from pi_ai.types import Context, Model, user_text

m = Model(id="x", provider="starlight", api="openai-completions", name="x", baseUrl="https://api.starlight.ai/v1")
print(build_params(m, Context(system_prompt="s", messages=[user_text("hi")]), {}))
```

---

# 路线 A：协议兼容现有的五种

## A1. 收集四样信息

1. **base_url**：注意各协议的约定不同。completions / responses 要带 `/v1`，Anthropic 只到域名（适配器自己补 `/v1/messages`），Google 要带 `/v1beta`。
2. **鉴权**：key 放哪个 header、从哪个环境变量读。
3. **模型清单**：每个模型的 id、上下文窗口、输出上限、价格（美元 / 百万 token）、是否支持思考。
4. **厂商 id**：自己起，小写连字符，比如 `starlight`。这个字符串会写进每个 `Model.provider`，也会成为 provider 的注册键，之后不好改。

## A2. 写模型目录

新建 `my_catalog/starlight.json`。**文件名必须等于厂商 id**，因为 `load_provider_models(provider_id)` 拿它拼路径。

```json
{
  "openai-completions": {
    "starlight-large": {
      "name": "Starlight Large",
      "baseUrl": "https://api.starlight.ai/v1",
      "reasoning": true,
      "input": ["text", "image"],
      "cost": { "input": 2.0, "output": 8.0, "cacheRead": 0.2, "cacheWrite": 0 },
      "contextWindow": 200000,
      "maxTokens": 32000,
      "thinkingLevelMap": {
        "off": null, "minimal": "low", "low": "low",
        "medium": "medium", "high": "high", "xhigh": null, "max": null
      }
    }
  }
}
```

几个要点：

- **最外层的 key 是 `api`（协议），不是厂商。**一家支持多协议就写多个 key，模型分别挂在下面。
- `id` / `provider` / `api` 三个字段**不要写在 json 里**。`flatten_model_catalog` 会从内层 key、传入的 provider_id、外层 key 分别补上。
- `cost` 的单位是美元每百万 token，`calculate_cost` 直接按它算钱。不关心费用就全填 0。
- `thinkingLevelMap` 把统一的七档级别映射成厂商自己的取值，`null` 表示不支持该档。请求了不支持的档位会被 `clamp_thinking_level` 压到最近的可用档，不报错。
- 字段名用 camelCase（线上 alias），和 `src/pi_ai/data/*.json` 保持一致。

## A3. 组装 provider

```python
from pathlib import Path

from pi_ai import create_models, create_provider
from pi_ai.api.openai_completions import openai_completions_api
from pi_ai.auth.env import env_api_key_auth
from pi_ai.catalog import load_provider_models

CATALOG_DIR = Path(__file__).parent / "my_catalog"


def starlight_provider():
    models = list(load_provider_models("starlight", CATALOG_DIR).values())
    return create_provider(
        id="starlight",                       # 必须等于 Model.provider
        name="Starlight",
        base_url=models[0].base_url,          # 元数据，实际请求看 model.base_url
        auth={"api_key": env_api_key_auth("Starlight API key", ["STARLIGHT_API_KEY"])},
        models=models,
        # 传成字典，model.api 对不上时会明确报错；传单个实现则不校验，静默用它
        api={"openai-completions": openai_completions_api()},
    )


models = create_models()
models.set_provider(starlight_provider())
model = models.get_model("starlight", "starlight-large")
```

四个容易搞混的地方：

- `create_provider(id=...)` **必须**等于目录里的 `Model.provider`，否则请求时报 `no provider registered for 'starlight'`。路由是拿 `model.provider` 反查注册表的。
- `create_provider(base_url=...)` 适配器**不读**，只是元数据。真正决定请求地址的是 `model.base_url`，或者 per-call 传的 `base_url` 选项。
- `api=` 传单个实现时，`Provider.streams_for()` 直接返回它，**不检查 `model.api`**。写错了不会报错，会拿错误的翻译器发请求。所以推荐用字典形式。
- `env_api_key_auth` 的第二个参数是**列表**，按顺序取第一个有值的环境变量，可以拿来做兼容回退。

## A4. 验证，不用真花钱

用 `respx` 假装上游，跑通再说：

```python
import asyncio, json, httpx, respx
from pi_ai import Context, user_text

SSE = (
    'data: {"choices":[{"delta":{"content":"你好"},"index":0}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],'
    '"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n'
    "data: [DONE]\n\n"
)


@respx.mock
async def main():
    seen = []

    def handler(request):
        seen.append((str(request.url), dict(request.headers), json.loads(request.content)))
        return httpx.Response(200, text=SSE)

    respx.post(url__regex=r".*").mock(side_effect=handler)

    reply = await models.complete_simple(model, Context(messages=[user_text("hi")]), api_key="sk-test")

    print("URL:", seen[0][0])
    print("请求体:", json.dumps(seen[0][2], ensure_ascii=False))
    print("回复:", reply.content, reply.stop_reason, reply.usage.cost.total)


asyncio.run(main())
```

对着打印出来的请求体和厂商文档逐字段核对。**任何情况下都不会抛异常**，出问题看 `reply.stop_reason == "error"` 和 `reply.error_message`。

## A5. 有差异就用 compat 抹平

协议对上了不代表字段完全一样。同为 `openai-completions`，各家的差异实测长这样：

```
openai       系统提示角色=developer   输出上限字段=max_completion_tokens   发 store
deepseek     系统提示角色=system      输出上限字段=max_tokens              不发 store
together     系统提示角色=system      输出上限字段=max_tokens              不发 store
```

这些差异有两个来源。第一个是 `detect_compat()`（`src/pi_ai/api/openai_completions_compat.py`），它按 provider 名和 baseUrl 认人，推断出一套基线——**你的新 id 它不认识，会落到最保守的默认值**。第二个是目录里显式写的 `compat`，逐字段覆盖推断值：

```json
"compat": {
  "maxTokensField": "max_tokens",
  "supportsStore": false,
  "supportsDeveloperRole": false,
  "thinkingFormat": "deepseek"
}
```

可用的开关看 `src/pi_ai/types.py` 里对应协议的 compat 类：`OpenAICompletionsCompat` 有 20 多个，`OpenAIResponsesCompat` 和 `AnthropicMessagesCompat` 各 8 个左右。

> **坑**：compat 类是 `extra="allow"`，**开关名写错不报错，会被静默忽略**。把 `supportsStrictMode` 少写一个 s，校验照样通过，但设置不生效。名字务必对着 `types.py` 里的 alias 抄。

注意 `openai-responses` 和 `openai-completions` 在这点上不对称：completions 有一整套推断基线，而 responses 几乎不推断，`supports_strict_mode`、`supports_tool_search` 这些默认全是 `False`，能力必须在目录里显式声明。

到这里路线 A 就完了。厂商如果同时提供第二种协议，在目录 json 里加一个顶层 key，再往 `api={...}` 字典里加一项即可，同一个 provider 会按 `model.api` 分派。

---

# 路线 B：全新协议，要写适配器

只有当上游的请求/响应结构和现有五种都对不上时才走这条。**适配器不需要在框架里注册**，实现一个满足 `ProviderStreams` 协议的对象直接传给 `create_provider(api=...)` 就行，可以完全放在你自己的项目里。

## B1. 先把协议摸清楚

动手前把这份清单填满，缺一项后面都要返工：

- 请求 URL 怎么拼，模型 id 在路径里还是 body 里
- 鉴权头的名字（`Authorization: Bearer`？`x-api-key`？）
- 请求体：消息列表的字段名和形状、系统提示放哪、工具 schema 的写法、输出上限字段名、思考参数怎么传
- 流式响应：是不是 SSE，事件名有哪些，文本增量在哪个字段
- 工具调用：是一次性给全还是流式拼参数，调用 id 叫什么
- 结束信号：finish reason 的取值有哪些
- 用量统计：字段名，以及**输出 token 含不含思考 token**（各家不一样）
- 错误响应的形状

## B2. 文件布局

照着现有适配器的约定拆三个文件（`google_generative_ai*` 是最短的模板，一共 400 行）：

```
starlight_chat_params.py   # Context -> 上游请求体
starlight_chat_stream.py   # 上游 chunk -> 统一事件（解析器）
starlight_chat.py          # 组装：URL、请求头、stream()、stream_simple()、导出工厂
```

## B3. 写 `build_params`：把 Context 翻译成请求体

```python
def build_params(model: Model, context: Context, options: Mapping[str, Any]) -> dict[str, Any]:
    messages = []
    if context.system_prompt:
        messages.append({"role": "system", "text": context.system_prompt})

    for msg in context.messages:
        if msg.role == "user":
            messages.append({"role": "user", "text": _text_of(msg)})
        elif msg.role == "assistant":
            messages.append(_assistant_to_wire(msg))
        elif msg.role == "toolResult":
            messages.append({"role": "tool", "call_id": msg.tool_call_id, "text": _text_of(msg)})

    params = {"model": model.id, "messages": messages, "stream": True}

    if context.tools:
        params["tools"] = [
            {"name": t.name, "desc": t.description, "schema": t.parameters} for t in context.tools
        ]
    if options.get("max_tokens"):
        params["max_output_tokens"] = options["max_tokens"]
    if options.get("temperature") is not None:
        params["temperature"] = options["temperature"]
    # 放最后，让调用方能覆盖上面任何字段
    if options.get("sampling_params"):
        params.update(options["sampling_params"])
    return params
```

三条必须遵守的：

- **助手消息要能原样回传。** `AssistantMessage.content` 里的 `ThinkingContent` 带着签名（`thinking_signature`），多轮对话必须把它发回去，推理链才不断。跨模型搬历史时签名无效，要降级成普通文本——现成的 `transform_messages()` 就是干这个的，直接复用。
- **工具结果靠 id 配对**，`ToolResultMessage.tool_call_id` 必须能对上你在解析阶段给 `ToolCall.id` 赋的值。
- `options` 里能拿到什么，看 `src/pi_ai/api/simple_options.py` 的 `_PASSTHROUGH_KEYS` 和 `_OPTIONAL_KEYS`。

## B4. 写解析器：上游 chunk 变成统一事件

这是最容易写错的一层，因为**事件序列是有契约的**：

```
start
  ├─ text_start → text_delta* → text_end
  ├─ thinking_start → thinking_delta* → thinking_end
  └─ toolcall_start → toolcall_delta* → toolcall_end
done  或  error
```

规则：

1. `content_index` 是这个块在 `output.content` 列表里的下标。
2. 每个 `*_start` 必须有配对的 `*_end`，流中断时在 `finish()` 里补发。
3. 每个事件都带 `partial`，即到此刻为止拼好的 `AssistantMessage`。**不用自己做快照**，`event_stream.push()` 会冻结它。
4. 结束时 `output.stop_reason` 必须从 `pending` 变成别的值，取值：`stop` / `length` / `toolUse` / `error` / `aborted` / `deferred`。有工具调用就是 `toolUse`。下面骨架里的 `_map_stop_reason()` 要你自己写，把厂商的 finish reason 字符串映射到这六个之一。
5. 拿到用量后填 `Usage` 再调 `calculate_cost(model, usage)`，费用就算好了。

骨架：

```python
@dataclass
class StarlightStreamParser:
    model: Model

    def __post_init__(self):
        self.output = AssistantMessage(
            api=self.model.api, provider=self.model.provider,
            model=self.model.id, stop_reason="pending",
        )
        self._current = None

    def feed(self, chunk: Mapping[str, Any]) -> Iterator[AssistantMessageEvent]:
        if delta := chunk.get("text_delta"):
            if self._current is None or self._current.type != "text":
                yield from self._close()
                self._current = TextContent(text="")
                self.output.content.append(self._current)
                yield TextStartEvent(content_index=self._index(), partial=self.output)
            self._current.text += delta
            yield TextDeltaEvent(content_index=self._index(), delta=delta, partial=self.output)

        if call := chunk.get("tool_call"):
            yield from self._close()
            tc = ToolCall(id=call["id"], name=call["name"], arguments=call.get("args") or {})
            self.output.content.append(tc)
            i = self._index()
            yield ToolCallStartEvent(content_index=i, partial=self.output)
            yield ToolCallDeltaEvent(content_index=i, delta=json.dumps(tc.arguments), partial=self.output)
            yield ToolCallEndEvent(content_index=i, tool_call=tc, partial=self.output)

        if reason := chunk.get("finish_reason"):
            self.output.raw_stop_reason = reason
            self.output.stop_reason = "toolUse" if any(
                b.type == "toolCall" for b in self.output.content
            ) else _map_stop_reason(reason)

        if u := chunk.get("usage"):
            self.output.usage = Usage(input=u["in"], output=u["out"], total_tokens=u["in"] + u["out"])
            calculate_cost(self.model, self.output.usage)

    def _index(self) -> int:
        return len(self.output.content) - 1

    def _close(self) -> Iterator[AssistantMessageEvent]:
        if self._current is None:
            return
        i = self.output.content.index(self._current)
        if self._current.type == "text":
            yield TextEndEvent(content_index=i, content=self._current.text, partial=self.output)
        else:
            yield ThinkingEndEvent(content_index=i, content=self._current.thinking, partial=self.output)
        self._current = None

    def finish(self) -> Iterator[AssistantMessageEvent]:
        yield from self._close()
```

## B5. 写 `stream()`：连接、消费、收尾

这一层几乎是纯样板，可以照抄，只改 URL 和请求头：

```python
def stream(model, context, options=None) -> AssistantMessageEventStream:
    options = dict(options or {})
    event_stream = AssistantMessageEventStream()

    async def run() -> None:
        parser = StarlightStreamParser(model)
        try:
            api_key = options.get("api_key")
            if not api_key:
                raise ProviderHTTPError(f"No API key for provider: {model.provider}")

            body = build_params(model, context, options)
            if (on_payload := options.get("on_payload")) is not None:
                replaced = on_payload(body, model)
                if asyncio.iscoroutine(replaced):
                    replaced = await replaced
                if replaced is not None:
                    body = replaced

            headers = merge_headers(
                {"content-type": "application/json", "authorization": f"Bearer {api_key}"},
                model.headers,
                options.get("headers"),
            )
            url = f"{(options.get('base_url') or model.base_url).rstrip('/')}/chat"

            await _consume(event_stream, parser, model, url, headers, body, options)
        except asyncio.CancelledError:
            _emit_error(event_stream, parser, aborted=True, error=None)
            raise
        except BaseException as error:      # 契约要求不抛出
            _emit_error(event_stream, parser, aborted=False, error=error)

    task = asyncio.ensure_future(run())
    event_stream.bind_producer(task)        # 让 cancel() 能中断真正的请求
    return event_stream


async def _consume(event_stream, parser, model, url, headers, body, options):
    async def connect():
        return await open_stream_post(
            url=url, headers=headers, json_body=body, timeout_ms=options.get("timeout_ms")
        )

    # 重试只包住建连；响应体一旦开始流就不能重放了
    opened = await retry_provider_request(
        connect,
        max_retries=options.get("max_retries") or 0,
        max_retry_delay_ms=options.get("max_retry_delay_ms"),
    )

    async with opened:
        if (on_response := options.get("on_response")) is not None:
            result = on_response(opened.response, model)
            if asyncio.iscoroutine(result):
                await result

        event_stream.push(StartEvent(partial=parser.output))

        async for sse in iter_sse(opened.body):
            data = sse.data.strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            for event in parser.feed(payload):
                event_stream.push(event)

    for event in parser.finish():
        event_stream.push(event)

    output = parser.output
    if output.stop_reason == "pending":
        raise ProviderHTTPError("stream ended without a finish reason")

    event_stream.push(DoneEvent(reason=output.stop_reason, message=output))
    event_stream.end()


def _emit_error(event_stream, parser, *, aborted: bool, error: BaseException | None) -> None:
    output = parser.output
    output.stop_reason = "aborted" if aborted else "error"
    output.error_message = (
        "Request was aborted" if error is None
        else format_provider_error(normalize_provider_error(error))
    )
    event_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
    event_stream.end()
```

需要的 import：

```python
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.http.client import merge_headers, open_stream_post
from pi_ai.http.errors import ProviderHTTPError, format_provider_error, normalize_provider_error
from pi_ai.http.retry import retry_provider_request
from pi_ai.http.sse import iter_sse
from pi_ai.cost import calculate_cost, clamp_thinking_level
from pi_ai.api.simple_options import build_base_options
```

## B6. 写 `stream_simple()`：把统一选项翻译成厂商参数

`stream` 是原始接口，选项直接透传；`stream_simple` 负责把 `reasoning="high"` 这种统一说法翻译成厂商自己的参数，并把输出上限压进上下文窗口。日常调用走的都是它。

```python
def stream_simple(model, context, options=None) -> AssistantMessageEventStream:
    options = dict(options or {})
    base = build_base_options(model, context, options, options.get("api_key"))

    reasoning = options.get("reasoning")
    if not reasoning:
        return stream(model, context, base)

    level = clamp_thinking_level(model, reasoning)   # 压到模型支持的最近一档
    if level == "off":
        return stream(model, context, base)
    return stream(model, context, {**base, "thinking": {"level": level}})
```

`build_base_options` 已经帮你算好 `max_tokens`、合并好 `sampling_params`。注意它**总会**给出 `max_tokens`：调用方没传就用 `model.max_tokens`，再用 `clamp_max_tokens_to_context` 压进上下文窗口剩余空间。所以走 `stream_simple` 时请求体里一定带输出上限，这是有意的。

## B7. 打包并注册

```python
class StarlightChatStreams:
    def stream(self, model, context, options): return stream(model, context, options)
    def stream_simple(self, model, context, options): return stream_simple(model, context, options)


def starlight_chat_api() -> StarlightChatStreams:
    return StarlightChatStreams()
```

`ProviderStreams` 是 `Protocol`，**不需要继承**，方法签名对上就行。然后目录 json 的顶层 key 用你的新协议 id：

```json
{ "starlight-chat": { "starlight-large": { "...": "..." } } }
```

```python
create_provider(
    id="starlight",
    name="Starlight",
    base_url="https://api.starlight.ai/v1",
    auth={"api_key": env_api_key_auth("Starlight API key", ["STARLIGHT_API_KEY"])},
    models=list(load_provider_models("starlight", CATALOG_DIR).values()),
    api={"starlight-chat": starlight_chat_api()},
)
```

## B8. 想让新协议支持 compat，要动一处框架代码

`Model` 的校验器按 `api` 查 `_COMPAT_BY_API`（`src/pi_ai/types.py`）决定用哪个 compat 类。**新 api id 不在表里，目录里写了 `compat` 会直接报错**：`api 'starlight-chat' does not support compat settings`。

需要开关的话，在 `types.py` 里加一个类并登记：

```python
class StarlightChatCompat(_Wire):
    supports_tool_streaming: bool | None = Field(default=None, alias="supportsToolStreaming")

_COMPAT_BY_API = {
    ...,
    "starlight-chat": StarlightChatCompat,
}
```

不需要开关就跳过这步，别在 json 里写 `compat` 就行。

## B9. 写测试

测试不打真实网络，用 `respx` 拦 httpx。`tests/conftest.py` 里有现成的工具：`gated_response()` 分段吐响应体，配合 `read_until()` 可以断言「事件是边到边推的，不是攒完才推」——这是流式实现最容易退化的地方。

至少覆盖这几条：正常文本流、工具调用流、HTTP 4xx 变成 `stop_reason == "error"`、传输层异常同样不抛、取消后 `stop_reason == "aborted"` 且已收内容保留、用量和费用算对。照着 `tests/test_openai_responses.py` 的结构写最快。

---

## 通用契约清单

不管走哪条路线，这几条都不能破：

1. **不抛异常。** 网络错误、4xx/5xx、未配置的厂商、超时，全部变成 `stop_reason == "error"` 的 `AssistantMessage`。调用方的流式渲染代码不套 try/except。
2. **事件顺序。** `start` 打头，每个 `*_start` 有配对的 `*_end`，`done` 或 `error` 收尾且只出现一次。
3. **`partial` 是快照。** 用 `push()` 推事件（它会冻结），不要用 `relay()`——那个是给转发已冻结事件用的。
4. **`stop_reason` 必须落定。** 停在 `pending` 说明流没正常结束，应该报错。
5. **取消要真的断开连接。** 用 `bind_producer(task)`，`cancel()` 才能中断 HTTP 请求让上游停止计费。
6. **用量算完要调 `calculate_cost`。** 分档定价、缓存价格都在里面处理。

## 常见坑

- **provider id 与 `Model.provider` 不一致** → `no provider registered for 'xxx'`，请求根本发不出去。
- **`api=` 传单个实现而 `model.api` 对不上** → 不报错，静默用错误的适配器，症状是 404 或解析不出内容。用字典形式可避免。
- **compat 开关名拼错** → 静默忽略。
- **给新协议写 compat 但没登记 compat 类** → 构造 `Model` 时就报错。
- **base_url 尾部斜杠和版本段** → 各协议约定不同，`build_request_url` 里都 `rstrip('/')` 了，但版本段是否包含要看协议。
- **输出 token 含不含思考 token** → Google 的 `candidatesTokenCount` 不含，要自己加 `thoughtsTokenCount`。新厂商务必查清楚，否则费用算错。
- **目录 json 手改了 `src/pi_ai/data/` 下的文件** → 那些是上游 TypeScript 项目生成的，`python scripts/sync_model_data.py` 会覆盖。自己的目录放自己的目录里。
