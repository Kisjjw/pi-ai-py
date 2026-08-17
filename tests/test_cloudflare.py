"""Cloudflare 认证解析与端点占位符替换。"""

import pytest

from pi_ai.auth.env import DefaultAuthContext
from pi_ai.auth.types import ApiKeyCredential
from pi_ai.providers._cloudflare import (
    CloudflareAIGatewayAuth,
    CloudflareWorkersAIAuth,
    resolve_cloudflare_model,
)
from pi_ai.types import Model

GATEWAY_URL = "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic"
WORKERS_URL = "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in ("CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_GATEWAY_ID"):
        monkeypatch.delenv(name, raising=False)


def model(base_url: str) -> Model:
    return Model(
        id="m",
        provider="cloudflare-ai-gateway",
        api="anthropic-messages",
        name="M",
        baseUrl=base_url,
        contextWindow=1000,
        maxTokens=100,
        cost={"input": 1, "output": 1},
    )


# --- 端点替换 ---


def test_placeholders_are_replaced_from_env():
    resolved = resolve_cloudflare_model(
        model(GATEWAY_URL),
        {"CLOUDFLARE_ACCOUNT_ID": "acct-1", "CLOUDFLARE_GATEWAY_ID": "gw-1"},
    )
    assert resolved.base_url == "https://gateway.ai.cloudflare.com/v1/acct-1/gw-1/anthropic"


def test_missing_env_key_keeps_the_placeholder():
    """留占位符而不是替换成空串：后者会拼出一个看似合法却指向错误资源的 URL。"""
    resolved = resolve_cloudflare_model(model(GATEWAY_URL), {"CLOUDFLARE_ACCOUNT_ID": "acct-1"})
    assert resolved.base_url == (
        "https://gateway.ai.cloudflare.com/v1/acct-1/{CLOUDFLARE_GATEWAY_ID}/anthropic"
    )


def test_no_env_returns_the_same_object():
    original = model(GATEWAY_URL)
    assert resolve_cloudflare_model(original, None) is original
    assert resolve_cloudflare_model(original, {}) is original


def test_url_without_placeholders_returns_the_same_object():
    original = model("https://example.com/v1")
    assert resolve_cloudflare_model(original, {"CLOUDFLARE_ACCOUNT_ID": "acct-1"}) is original


def test_replacement_does_not_touch_other_model_fields():
    resolved = resolve_cloudflare_model(model(WORKERS_URL), {"CLOUDFLARE_ACCOUNT_ID": "acct-1"})
    assert resolved.id == "m"
    assert resolved.context_window == 1000
    assert resolved.cost.input == 1


# --- workers-ai 认证 ---


async def test_workers_ai_resolves_from_env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")

    result = await CloudflareWorkersAIAuth().resolve(ctx=DefaultAuthContext(), credential=None)
    assert result is not None
    assert result.auth.api_key == "cf-key"
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "acct-1"}
    assert result.source == "CLOUDFLARE_API_KEY"


async def test_workers_ai_needs_both_key_and_account(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    assert await CloudflareWorkersAIAuth().resolve(ctx=DefaultAuthContext(), credential=None) is None


async def test_workers_ai_ignores_gateway_id(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gw-1")

    result = await CloudflareWorkersAIAuth().resolve(ctx=DefaultAuthContext(), credential=None)
    assert result is not None
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "acct-1"}


# --- ai-gateway 认证 ---


async def test_ai_gateway_uses_the_cf_aig_header(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gw-1")

    result = await CloudflareAIGatewayAuth().resolve(ctx=DefaultAuthContext(), credential=None)
    assert result is not None
    assert result.auth.api_key is None
    assert result.auth.headers == {
        "cf-aig-authorization": "Bearer cf-key",
        "Authorization": None,
        "x-api-key": None,
    }
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "acct-1", "CLOUDFLARE_GATEWAY_ID": "gw-1"}


async def test_ai_gateway_needs_the_gateway_id(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-1")
    assert await CloudflareAIGatewayAuth().resolve(ctx=DefaultAuthContext(), credential=None) is None


# --- 逐字段合并 ---


async def test_credential_with_only_a_key_picks_up_account_from_env(monkeypatch):
    """凭据只存了 API key 时，账号 id 仍要从环境变量补齐。"""
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-env")

    result = await CloudflareWorkersAIAuth().resolve(
        ctx=DefaultAuthContext(),
        credential=ApiKeyCredential(key="cred-key"),
    )
    assert result is not None
    assert result.auth.api_key == "cred-key"
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "acct-env"}
    assert result.source == "stored credential"


async def test_credential_values_win_over_env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "env-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-env")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gw-env")

    result = await CloudflareAIGatewayAuth().resolve(
        ctx=DefaultAuthContext(),
        credential=ApiKeyCredential(
            key="cred-key",
            env={"CLOUDFLARE_ACCOUNT_ID": "acct-cred", "CLOUDFLARE_GATEWAY_ID": "gw-cred"},
        ),
    )
    assert result is not None
    assert result.auth.headers["cf-aig-authorization"] == "Bearer cred-key"
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "acct-cred", "CLOUDFLARE_GATEWAY_ID": "gw-cred"}


async def test_partial_credential_env_merges_per_field(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "env-key")
    monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "gw-env")

    result = await CloudflareAIGatewayAuth().resolve(
        ctx=DefaultAuthContext(),
        credential=ApiKeyCredential(key=None, env={"CLOUDFLARE_ACCOUNT_ID": "acct-cred"}),
    )
    assert result is not None
    # key 从环境变量来，账号从凭据来，网关从环境变量来
    assert result.auth.headers["cf-aig-authorization"] == "Bearer env-key"
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "acct-cred", "CLOUDFLARE_GATEWAY_ID": "gw-env"}


async def test_scoped_env_overrides_process_env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-process")
    ctx = DefaultAuthContext({"CLOUDFLARE_API_KEY": "scoped-key", "CLOUDFLARE_ACCOUNT_ID": "acct-scoped"})

    result = await CloudflareWorkersAIAuth().resolve(ctx=ctx, credential=None)
    assert result is not None
    assert result.auth.api_key == "scoped-key"
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "acct-scoped"}
