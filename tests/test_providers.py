import pytest

from pi_ai.providers import (
    PROVIDER_SPECS,
    deepseek_provider,
    openai_completions_providers,
    openrouter_provider,
    provider_by_id,
)
from pi_ai.registry import create_models

ALL_ENV_VARS = sorted({var for spec in PROVIDER_SPECS for var in spec.env_vars})


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """避免开发机上真实设置的 key 影响断言。"""
    for var in ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_the_table_covers_nineteen_providers():
    assert len(PROVIDER_SPECS) == 19
    assert len({spec.id for spec in PROVIDER_SPECS}) == 19


@pytest.mark.parametrize("spec", PROVIDER_SPECS, ids=lambda spec: spec.id)
def test_every_provider_loads_its_catalog(spec):
    provider = provider_by_id(spec.id)
    models = provider.get_models()
    assert models, f"{spec.id} has no models"
    assert all(m.api == "openai-completions" for m in models)
    assert all(m.provider == spec.id for m in models)


@pytest.mark.parametrize("spec", PROVIDER_SPECS, ids=lambda spec: spec.id)
async def test_every_provider_resolves_its_documented_env_var(spec, monkeypatch):
    monkeypatch.setenv(spec.env_vars[0], "sk-test")
    models = create_models()
    models.set_provider(provider_by_id(spec.id))

    resolved = await models.get_auth(spec.id)
    assert resolved is not None
    assert resolved.auth.api_key == "sk-test"
    assert resolved.source == spec.env_vars[0]


@pytest.mark.parametrize("spec", PROVIDER_SPECS, ids=lambda spec: spec.id)
async def test_every_provider_is_unconfigured_without_its_env_var(spec):
    models = create_models()
    models.set_provider(provider_by_id(spec.id))
    assert await models.get_auth(spec.id) is None


def test_all_providers_can_share_one_collection():
    providers = openai_completions_providers()
    models = create_models()
    for provider in providers:
        models.set_provider(provider)

    assert len(models.get_providers()) == len(PROVIDER_SPECS)
    total = sum(len(p.get_models()) for p in providers)
    assert len(models.get_models()) == total
    assert total > 200


def test_base_urls_match_the_catalog_entries():
    for spec in PROVIDER_SPECS:
        provider = provider_by_id(spec.id)
        for model in provider.get_models():
            assert model.base_url, f"{spec.id}/{model.id} has no base_url"


def test_unknown_provider_id_raises():
    with pytest.raises(KeyError):
        provider_by_id("nope")


def test_moonshot_variants_share_one_env_var():
    cn = provider_by_id("moonshotai-cn")
    intl = provider_by_id("moonshotai")
    assert cn.auth["api_key"].env_vars == intl.auth["api_key"].env_vars == ["MOONSHOT_API_KEY"]
    assert cn.base_url != intl.base_url


def test_qwen_individual_shares_the_international_endpoint_and_key():
    plan = provider_by_id("qwen-token-plan")
    individual = provider_by_id("qwen-token-plan-individual")
    assert plan.base_url == individual.base_url
    assert plan.auth["api_key"].env_vars == individual.auth["api_key"].env_vars
    # 但 Individual 的模型目录更窄
    assert len(individual.get_models()) < len(plan.get_models())


def test_named_factories_match_provider_by_id():
    assert deepseek_provider().id == "deepseek"
    assert openrouter_provider().id == "openrouter"
