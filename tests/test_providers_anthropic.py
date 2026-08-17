import pytest

from pi_ai.providers import (
    ANTHROPIC_COMPATIBLE_SPECS,
    anthropic_messages_providers,
    anthropic_provider,
    provider_by_id,
)
from pi_ai.providers._anthropic_table import (
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_AUTH_TOKEN_ENV,
    ANTHROPIC_OAUTH_TOKEN_ENV,
)
from pi_ai.registry import create_models

ALL_ENV_VARS = sorted(
    {var for spec in ANTHROPIC_COMPATIBLE_SPECS for var in spec.env_vars}
    | {ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV}
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for var in ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_five_anthropic_providers_are_available():
    providers = anthropic_messages_providers()
    assert [p.id for p in providers] == [
        "anthropic",
        "kimi-coding",
        "minimax",
        "minimax-cn",
        "vercel-ai-gateway",
    ]


@pytest.mark.parametrize(
    "provider_id", ["anthropic", "kimi-coding", "minimax", "minimax-cn", "vercel-ai-gateway"]
)
def test_every_provider_loads_anthropic_models(provider_id):
    provider = provider_by_id(provider_id)
    models = provider.get_models()
    assert models
    assert all(m.api == "anthropic-messages" for m in models)
    assert all(m.provider == provider_id for m in models)


@pytest.mark.parametrize("spec", ANTHROPIC_COMPATIBLE_SPECS, ids=lambda s: s.id)
async def test_compatible_providers_resolve_their_env_var(spec, monkeypatch):
    monkeypatch.setenv(spec.env_vars[0], "sk-test")
    models = create_models()
    models.set_provider(provider_by_id(spec.id))

    resolved = await models.get_auth(spec.id)
    assert resolved is not None
    assert resolved.auth.api_key == "sk-test"
    assert resolved.source == spec.env_vars[0]


@pytest.mark.parametrize("spec", ANTHROPIC_COMPATIBLE_SPECS, ids=lambda s: s.id)
async def test_compatible_providers_are_unconfigured_without_env(spec):
    models = create_models()
    models.set_provider(provider_by_id(spec.id))
    assert await models.get_auth(spec.id) is None


# --- Anthropic 的自定义解析顺序 ---


async def test_anthropic_resolves_the_api_key_env(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "sk-ant-api03-x")
    models = create_models()
    models.set_provider(anthropic_provider())

    resolved = await models.get_auth("anthropic")
    assert resolved is not None
    assert resolved.auth.api_key == "sk-ant-api03-x"
    assert resolved.source == ANTHROPIC_API_KEY_ENV


async def test_oauth_token_env_beats_the_api_key(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "sk-ant-api03-x")
    monkeypatch.setenv(ANTHROPIC_OAUTH_TOKEN_ENV, "sk-ant-oat01-y")
    models = create_models()
    models.set_provider(anthropic_provider())

    resolved = await models.get_auth("anthropic")
    assert resolved is not None
    assert resolved.auth.api_key == "sk-ant-oat01-y"
    assert resolved.source == ANTHROPIC_OAUTH_TOKEN_ENV


async def test_auth_token_env_wins_and_uses_the_header_channel(monkeypatch):
    """ANTHROPIC_AUTH_TOKEN 返回的是 header，不是 api_key。"""
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "sk-ant-api03-x")
    monkeypatch.setenv(ANTHROPIC_OAUTH_TOKEN_ENV, "sk-ant-oat01-y")
    monkeypatch.setenv(ANTHROPIC_AUTH_TOKEN_ENV, "bearer-z")
    models = create_models()
    models.set_provider(anthropic_provider())

    resolved = await models.get_auth("anthropic")
    assert resolved is not None
    assert resolved.auth.api_key is None
    assert resolved.auth.headers == {"Authorization": "Bearer bearer-z"}
    assert resolved.source == ANTHROPIC_AUTH_TOKEN_ENV


async def test_stored_credential_beats_every_env_var(monkeypatch):
    from pi_ai.auth.store import InMemoryCredentialStore
    from pi_ai.auth.types import ApiKeyCredential

    monkeypatch.setenv(ANTHROPIC_AUTH_TOKEN_ENV, "bearer-z")
    store = InMemoryCredentialStore()

    async def put(_current):
        return ApiKeyCredential(key="sk-stored")

    await store.modify("anthropic", put)

    models = create_models(credentials=store)
    models.set_provider(anthropic_provider())

    resolved = await models.get_auth("anthropic")
    assert resolved is not None
    assert resolved.auth.api_key == "sk-stored"


async def test_anthropic_is_unconfigured_without_any_env():
    models = create_models()
    models.set_provider(anthropic_provider())
    assert await models.get_auth("anthropic") is None


def test_all_providers_can_share_one_collection():
    models = create_models()
    for provider in anthropic_messages_providers():
        models.set_provider(provider)
    assert len(models.get_providers()) == 5
    assert len(models.get_models()) > 20
