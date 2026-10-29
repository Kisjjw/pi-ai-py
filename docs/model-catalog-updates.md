# OpenAI、Anthropic、DeepSeek 模型目录更新

核对日期：2026-10-29。以下为这三家官方 API 的通用文本 / 图像模型；价格单位为美元 / 百万 tokens，采用标准档，不含 Batch、Fast、地区加价等额外倍率。现有旧模型继续保留。

| 厂商 | 模型 ID | 本次变更 | 输入 | 上下文 / 最大输出 | 输入 / 输出价格 |
| --- | --- | --- | --- | --- | --- |
| OpenAI | `gpt-6-astra` | 上下文窗口修正 | 文本、图像 | 1,050,000 / 128,000 | 10 / 50 |
| OpenAI | `gpt-6-sol` | 新增目录项 | 文本、图像 | 1,050,000 / 128,000 | 2 / 10 |
| OpenAI | `gpt-6.1-sol` | 新增目录项 | 文本、图像 | 1,050,000 / 128,000 | 2 / 10 |
| OpenAI | `gpt-6.1-terra` | 新增目录项 | 文本、图像 | 1,050,000 / 128,000 | 1 / 5 |
| OpenAI | `gpt-6-luna` | 新增目录项 | 文本、图像 | 1,050,000 / 128,000 | 0.1 / 0.5 |
| Anthropic | `claude-opus-5-5` | 新增目录项 | 文本、图像 | 1,000,000 / 128,000 | 4 / 20 |
| Anthropic | `claude-sonnet-5-5` | 新增目录项 | 文本、图像 | 1,000,000 / 128,000 | 2 / 10 |
| Anthropic | `claude-haiku-5-5` | 新增目录项 | 文本、图像 | 1,000,000 / 128,000 | 0.1 / 0.5 起 |
| DeepSeek | `deepseek-flash` | 展示版本更新为 V4.1 Flash 1017 | 文本、图像 | 1,000,000 / 384,000 | 0.3 / 1.2，高峰档 |
| DeepSeek | `deepseek-v4-pro` | 展示版本明确为 V4 Pro 0813 | 文本 | 1,000,000 / 384,000 | 1.32 / 3.96，高峰档 |

## 推理与协议

GPT 模型继续走 `openai-responses`。GPT-6 Astra 和 GPT-6.1 Sol 不支持关闭推理，`reasoning="off"` 会压到 `low`；其余表列 GPT 模型支持 `none`。这些模型均支持 `low`、`medium`、`high`、`xhigh`、`max`。

Claude 5.5 模型走 `anthropic-messages`，使用 adaptive thinking，通过 effort 控制推理强度，均支持 `low`、`medium`、`high`、`xhigh`、`max`。它们不接受旧的思考预算模式或采样温度参数。

- Opus 5.5 始终启用思考。省略 `reasoning` 时保留服务器默认；显式传 `off` 会压到 `low`。
- Sonnet 5.5 的最低思考模式是 `between_tools`，会关闭前置思考，但仍可能在工具调用之间思考；省略 `reasoning` 或传 `off` 时使用此模式，且不携带 `display` 或 `budget_tokens`。
- Haiku 5.5 支持 `disabled`，省略 `reasoning` 或传 `off` 时沿用客户端关闭思考的行为。

DeepSeek 本次没有新 API 模型 ID。官方已将 `deepseek-flash` 升级为 DeepSeek-V4.1-Flash-1017，调用方式和价格不变；Pro 继续使用 `deepseek-v4-pro`。保留现有 Chat Completions 接入。

## 计费边界

GPT 表列模型的 1,050,000 是上下文窗口，272,000 是长输入加价阈值，两者不同。输入总量超过 272,000 tokens 时，整次请求的输入、缓存读取和写入价格翻倍，输出价格乘以 1.5。GPT-6.1 Sol / Terra 的缓存读取为基础输入价的 5%，GPT-6 Sol / Luna / Astra 为 10%。缓存写入采用 30 分钟基础价格，即基础输入价的 1.25 倍。

Haiku 5.5 输入超过 100,000 tokens 后，整次请求改用输入 0.5、输出 2.5、缓存读取 0.05、5 分钟缓存写入 0.625 的价格。阈值计算包含普通输入、缓存读取、缓存写入，恰好 100,000 时仍使用基础价格。Opus 5.5 和 Sonnet 5.5 的缓存读取为基础输入价的 5%，5 分钟缓存写入为 1.25 倍。

DeepSeek 目录继续保存高峰档美元价格；官方空闲时段为该价格的一半。当前成本计算不会依据时段或中国法定节假日自动切价，因此空闲时段的 `reply.usage.cost` 是按高峰档估算的值。

## 使用示例

```python
from pi_ai import Context, create_models, user_text
from pi_ai.providers import provider_by_id

models = create_models()
for provider_id in ("openai", "anthropic", "deepseek"):
    models.set_provider(provider_by_id(provider_id))

model = models.get_model("openai", "gpt-6.1-sol")
assert model is not None
# 同样可选 ("anthropic", "claude-sonnet-5-5") 或 ("deepseek", "deepseek-flash")。
# 在 async 函数内调用；各家的 API key 按 README 配置。
reply = await models.complete_simple(
    model, Context(messages=[user_text("介绍一下你的能力")]), reasoning="medium"
)
```

## 官方来源与维护

- OpenAI：[模型总览](https://developers.openai.com/api/docs/models)、[Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)、[Sol](https://developers.openai.com/api/docs/models/gpt-6-sol)、[6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)、[6.1 Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)、[Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)。
- Anthropic：[Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/overview)、[Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/overview)、[Haiku 5.5](https://platform.claude.com/docs/en/models/haiku-5-5/overview)、[价格](https://platform.claude.com/docs/en/about-claude/pricing)、[Sonnet 迁移指南](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide)。
- DeepSeek：[模型与价格](https://api-docs.deepseek.com/quick_start/pricing/)、[1017 发布公告](https://api-docs.deepseek.com/news/news261017/)。

基础目录来自上游 TypeScript 项目，包内 `.manifest.json` 记录的是原始上游同步快照。本次根据官方文档对 `openai.json`、`anthropic.json`、`deepseek.json` 作定向修订；其他厂商目录不变。再次运行全量同步脚本会覆盖这三份文件，应先将本次修订合并至上游目录，或在同步后重新应用并运行目录与协议测试。
