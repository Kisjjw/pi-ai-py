import json

from pi_ai.auth.env import DefaultAuthContext, env_api_key_auth
from pi_ai.auth.store import FileCredentialStore, InMemoryCredentialStore
from pi_ai.auth.types import ApiKeyCredential


class FakeContext:
    def __init__(self, env: dict[str, str]) -> None:
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


def _returns(value):
    """构造一个忽略入参、直接返回 value 的 modify 回调。"""

    async def callback(_current):
        return value

    return callback


async def test_env_auth_resolves_from_first_set_env_var():
    auth = env_api_key_auth("Test key", ["FIRST_KEY", "SECOND_KEY"])
    result = await auth.resolve(ctx=FakeContext({"SECOND_KEY": "sk-2"}), credential=None)
    assert result is not None
    assert result.auth.api_key == "sk-2"
    assert result.source == "SECOND_KEY"


async def test_env_auth_prefers_first_listed_env_var():
    auth = env_api_key_auth("Test key", ["FIRST_KEY", "SECOND_KEY"])
    result = await auth.resolve(ctx=FakeContext({"FIRST_KEY": "sk-1", "SECOND_KEY": "sk-2"}), credential=None)
    assert result is not None
    assert result.auth.api_key == "sk-1"


async def test_stored_credential_beats_env_var():
    auth = env_api_key_auth("Test key", ["FIRST_KEY"])
    credential = ApiKeyCredential(key="sk-stored", env={"ACCOUNT": "a1"})
    result = await auth.resolve(ctx=FakeContext({"FIRST_KEY": "sk-env"}), credential=credential)
    assert result is not None
    assert result.auth.api_key == "sk-stored"
    assert result.env == {"ACCOUNT": "a1"}
    assert result.source == "stored credential"


async def test_unconfigured_provider_resolves_to_none():
    auth = env_api_key_auth("Test key", ["MISSING_KEY"])
    assert await auth.resolve(ctx=FakeContext({}), credential=None) is None


async def test_default_context_reads_process_env(monkeypatch):
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk-proc")
    ctx = DefaultAuthContext()
    assert await ctx.env("PI_AI_TEST_KEY") == "sk-proc"
    assert await ctx.env("PI_AI_TEST_ABSENT") is None


async def test_default_context_scoped_env_overrides_process_env(monkeypatch):
    monkeypatch.setenv("PI_AI_TEST_KEY", "sk-proc")
    ctx = DefaultAuthContext(env={"PI_AI_TEST_KEY": "sk-scoped"})
    assert await ctx.env("PI_AI_TEST_KEY") == "sk-scoped"


async def test_memory_store_round_trip():
    store = InMemoryCredentialStore()
    assert await store.read("openai") is None

    await store.modify("openai", _returns(ApiKeyCredential(key="sk-1")))
    stored = await store.read("openai")
    assert stored is not None
    assert stored.key == "sk-1"

    assert [info.provider_id for info in await store.list()] == ["openai"]

    await store.delete("openai")
    assert await store.read("openai") is None


async def test_memory_store_modify_sees_current_value():
    store = InMemoryCredentialStore()
    await store.modify("openai", _returns(ApiKeyCredential(key="sk-1")))

    seen: list[object] = []

    async def bump(current):
        seen.append(current)
        return ApiKeyCredential(key="sk-2")

    await store.modify("openai", bump)
    assert isinstance(seen[0], ApiKeyCredential)
    assert seen[0].key == "sk-1"


async def test_memory_store_modify_returning_none_leaves_entry_unchanged():
    store = InMemoryCredentialStore()
    await store.modify("openai", _returns(ApiKeyCredential(key="sk-1")))

    async def noop(current):
        return None

    await store.modify("openai", noop)
    stored = await store.read("openai")
    assert stored is not None
    assert stored.key == "sk-1"


async def test_file_store_persists_across_instances(tmp_path):
    path = tmp_path / "auth.json"
    store = FileCredentialStore(path)
    await store.modify("openai", _returns(ApiKeyCredential(key="sk-file")))

    reopened = FileCredentialStore(path)
    stored = await reopened.read("openai")
    assert stored is not None
    assert stored.key == "sk-file"
    assert json.loads(path.read_text(encoding="utf-8"))["openai"]["type"] == "api_key"


async def test_file_store_handles_missing_file(tmp_path):
    store = FileCredentialStore(tmp_path / "absent.json")
    assert await store.read("openai") is None
    assert await store.list() == []
