from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    """一家厂商的静态配置。

    对应 TS 的 src/providers/<id>.ts —— 那边每家就是 15 行的 createProvider 调用，
    这里合并成一张表。base_url 与模型目录来自打包进来的 data/<id>.json。
    """

    id: str
    name: str
    base_url: str
    auth_name: str
    env_vars: tuple[str, ...]


#: 走 openai-completions 协议且认证方式为标准环境变量的厂商。
#:
#: 不在此表中的同协议厂商：
#: - cloudflare-workers-ai：自定义认证（API key + 账号 id）与端点占位符替换，见 M6
#: - 混合协议厂商（xai、fireworks、github-copilot、openrouter、cloudflare-ai-gateway、
#:   opencode、opencode-go）：需要多个适配器就绪，见 M6
PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="ant-ling",
        name="Ant Ling",
        base_url="https://api.ant-ling.com/v1",
        auth_name="Ant Ling API key",
        env_vars=("ANT_LING_API_KEY",),
    ),
    ProviderSpec(
        id="baseten",
        name="Baseten",
        base_url="https://inference.baseten.co/v1",
        auth_name="Baseten API key",
        env_vars=("BASETEN_API_KEY",),
    ),
    ProviderSpec(
        id="cerebras",
        name="Cerebras",
        base_url="https://api.cerebras.ai/v1",
        auth_name="Cerebras API key",
        env_vars=("CEREBRAS_API_KEY",),
    ),
    ProviderSpec(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        auth_name="DeepSeek API key",
        env_vars=("DEEPSEEK_API_KEY",),
    ),
    ProviderSpec(
        id="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        auth_name="Groq API key",
        env_vars=("GROQ_API_KEY",),
    ),
    ProviderSpec(
        id="huggingface",
        name="Hugging Face",
        base_url="https://router.huggingface.co/v1",
        auth_name="Hugging Face token",
        env_vars=("HF_TOKEN",),
    ),
    ProviderSpec(
        id="moonshotai",
        name="Moonshot AI",
        base_url="https://api.moonshot.ai/v1",
        auth_name="Moonshot AI API key",
        env_vars=("MOONSHOT_API_KEY",),
    ),
    ProviderSpec(
        id="moonshotai-cn",
        name="Moonshot AI CN",
        base_url="https://api.moonshot.cn/v1",
        auth_name="Moonshot AI API key",
        env_vars=("MOONSHOT_API_KEY",),
    ),
    ProviderSpec(
        id="nvidia",
        name="NVIDIA",
        base_url="https://integrate.api.nvidia.com/v1",
        auth_name="NVIDIA API key",
        env_vars=("NVIDIA_API_KEY",),
    ),
    ProviderSpec(
        id="qwen-token-plan",
        name="Qwen Token Plan",
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        auth_name="Qwen Token Plan API key",
        env_vars=("QWEN_TOKEN_PLAN_API_KEY",),
    ),
    ProviderSpec(
        id="qwen-token-plan-cn",
        name="Qwen Token Plan CN",
        base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        auth_name="Qwen Token Plan CN API key",
        env_vars=("QWEN_TOKEN_PLAN_CN_API_KEY",),
    ),
    ProviderSpec(
        id="qwen-token-plan-individual",
        name="Qwen Token Plan Individual",
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        auth_name="Qwen Token Plan Individual API key",
        env_vars=("QWEN_TOKEN_PLAN_API_KEY",),
    ),
    ProviderSpec(
        id="together",
        name="Together",
        base_url="https://api.together.ai/v1",
        auth_name="Together API key",
        env_vars=("TOGETHER_API_KEY",),
    ),
    ProviderSpec(
        id="xiaomi",
        name="Xiaomi",
        base_url="https://api.xiaomimimo.com/v1",
        auth_name="Xiaomi API key",
        env_vars=("XIAOMI_API_KEY",),
    ),
    ProviderSpec(
        id="xiaomi-token-plan-cn",
        name="Xiaomi Token Plan CN",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        auth_name="Xiaomi Token Plan CN API key",
        env_vars=("XIAOMI_TOKEN_PLAN_CN_API_KEY",),
    ),
    ProviderSpec(
        id="xiaomi-token-plan-ams",
        name="Xiaomi Token Plan AMS",
        base_url="https://token-plan-ams.xiaomimimo.com/v1",
        auth_name="Xiaomi Token Plan AMS API key",
        env_vars=("XIAOMI_TOKEN_PLAN_AMS_API_KEY",),
    ),
    ProviderSpec(
        id="xiaomi-token-plan-sgp",
        name="Xiaomi Token Plan SGP",
        base_url="https://token-plan-sgp.xiaomimimo.com/v1",
        auth_name="Xiaomi Token Plan SGP API key",
        env_vars=("XIAOMI_TOKEN_PLAN_SGP_API_KEY",),
    ),
    ProviderSpec(
        id="zai",
        name="Z.AI",
        base_url="https://api.z.ai/api/coding/paas/v4",
        auth_name="Z.AI API key",
        env_vars=("ZAI_API_KEY",),
    ),
    ProviderSpec(
        id="zai-coding-cn",
        name="Z.AI Coding CN",
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        auth_name="Z.AI Coding CN API key",
        env_vars=("ZAI_CODING_CN_API_KEY",),
    ),
)

PROVIDER_SPECS_BY_ID = {spec.id: spec for spec in PROVIDER_SPECS}
