import json

import pytest

from pi_ai.catalog import flatten_model_catalog, load_provider_models


def test_flatten_merges_api_groups_and_fills_identity_fields():
    groups = {
        "openai-completions": {
            "grok-4": {
                "name": "Grok 4",
                "baseUrl": "https://api.x.ai/v1",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 3, "output": 15, "cacheRead": 0.75, "cacheWrite": 0},
                "contextWindow": 256000,
                "maxTokens": 32000,
            }
        },
        "openai-responses": {
            "grok-4-fast": {
                "name": "Grok 4 Fast",
                "baseUrl": "https://api.x.ai/v1",
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0.2, "output": 0.5, "cacheRead": 0.05, "cacheWrite": 0},
                "contextWindow": 2000000,
                "maxTokens": 30000,
            }
        },
    }

    models = flatten_model_catalog("xai", groups)

    assert set(models) == {"grok-4", "grok-4-fast"}
    assert models["grok-4"].id == "grok-4"
    assert models["grok-4"].provider == "xai"
    assert models["grok-4"].api == "openai-completions"
    assert models["grok-4"].context_window == 256000
    assert models["grok-4-fast"].api == "openai-responses"


def test_flatten_parses_compat_by_api():
    groups = {
        "openai-completions": {
            "m": {
                "name": "M",
                "baseUrl": "http://x/v1",
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 1000,
                "maxTokens": 100,
                "compat": {"thinkingFormat": "zai", "supportsStore": False},
            }
        }
    }
    models = flatten_model_catalog("p", groups)
    assert models["m"].compat.thinking_format == "zai"
    assert models["m"].compat.supports_store is False


def test_load_reads_provider_json(tmp_path):
    payload = {
        "openai-completions": {
            "m": {
                "name": "M",
                "baseUrl": "http://x/v1",
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 1000,
                "maxTokens": 100,
            }
        }
    }
    (tmp_path / "acme.json").write_text(json.dumps(payload), encoding="utf-8")

    models = load_provider_models("acme", data_dir=tmp_path)
    assert models["m"].provider == "acme"


def test_load_raises_for_unknown_provider(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_provider_models("nope", data_dir=tmp_path)
