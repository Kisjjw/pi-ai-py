import asyncio

from pi_ai.auth.env import env_api_key_auth
from pi_ai.auth.store import InMemoryCredentialStore
from pi_ai.auth.types import ApiKeyCredential
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.provider import create_provider
from pi_ai.registry import create_models
from pi_ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    user_text,
)


class RecordingStreams:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def _run(self, model, context, options):
        self.calls.append({"model": model, "context": context, "options": dict(options)})
        stream = AssistantMessageEventStream()
        stream.push(
            DoneEvent(
                reason="stop",
                message=AssistantMessage(
                    api=model.api, provider=model.provider, model=model.id, stop_reason="stop"
                ),
            )
        )
        return stream

    def stream(self, model, context, options):
        return self._run(model, context, options)

    def stream_simple(self, model, context, options):
        return self._run(model, context, options)


def make_model(model_id="m", api="openai-completions", headers=None) -> Model:
    return Model(
        id=model_id,
        name=model_id.upper(),
        api=api,
        provider="acme",
        base_url="http://acme.test/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=1000,
        max_tokens=100,
        headers=headers,
    )


def make_provider(streams, models=None, api=None):
    return create_provider(
        id="acme",
        name="Acme",
        base_url="http://acme.test/v1",
        auth={"api_key": env_api_key_auth("Acme key", ["ACME_API_KEY"])},
        models=models or [make_model()],
        api=api if api is not None else streams,
    )


async def test_registered_models_are_queryable():
    models = create_models()
    models.set_provider(make_provider(RecordingStreams(), models=[make_model("a"), make_model("b")]))

    assert [m.id for m in models.get_models()] == ["a", "b"]
    assert [m.id for m in models.get_models("acme")] == ["a", "b"]
    assert models.get_model("acme", "a").id == "a"
    assert models.get_model("acme", "missing") is None
    assert models.get_model("nope", "a") is None
    assert [p.id for p in models.get_providers()] == ["acme"]


async def test_setting_provider_twice_replaces_it():
    models = create_models()
    models.set_provider(make_provider(RecordingStreams(), models=[make_model("a")]))
    models.set_provider(make_provider(RecordingStreams(), models=[make_model("b")]))
    assert [m.id for m in models.get_models()] == ["b"]


async def test_auth_resolves_from_env(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "sk-env")
    models = create_models()
    models.set_provider(make_provider(RecordingStreams()))

    resolved = await models.get_auth("acme")
    assert resolved is not None
    assert resolved.auth.api_key == "sk-env"
    assert resolved.source == "ACME_API_KEY"


async def test_auth_prefers_stored_credential(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "sk-env")
    store = InMemoryCredentialStore()

    async def put(_current):
        return ApiKeyCredential(key="sk-stored")

    await store.modify("acme", put)

    models = create_models(credentials=store)
    models.set_provider(make_provider(RecordingStreams()))

    resolved = await models.get_auth("acme")
    assert resolved is not None
    assert resolved.auth.api_key == "sk-stored"


async def test_auth_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    models = create_models()
    models.set_provider(make_provider(RecordingStreams()))
    assert await models.get_auth("acme") is None


async def test_stream_injects_resolved_api_key(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "sk-env")
    streams = RecordingStreams()
    models = create_models()
    models.set_provider(make_provider(streams))

    stream = models.stream(models.get_model("acme", "m"), Context(messages=[user_text("hi")]))
    result = await stream.result()

    assert result.stop_reason == "stop"
    assert streams.calls[0]["options"]["api_key"] == "sk-env"


async def test_explicit_api_key_beats_resolved_auth(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "sk-env")
    streams = RecordingStreams()
    models = create_models()
    models.set_provider(make_provider(streams))

    await models.complete(
        models.get_model("acme", "m"),
        Context(messages=[user_text("hi")]),
        api_key="sk-explicit",
    )
    assert streams.calls[0]["options"]["api_key"] == "sk-explicit"


async def test_header_precedence_auth_then_model_then_explicit(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "sk-env")
    streams = RecordingStreams()
    models = create_models()
    models.set_provider(
        make_provider(streams, models=[make_model(headers={"x-model": "from-model", "x-both": "model"})])
    )

    await models.complete(
        models.get_model("acme", "m"),
        Context(messages=[user_text("hi")]),
        headers={"x-both": "explicit", "x-caller": "yes"},
    )

    sent = streams.calls[0]["options"]["headers"]
    assert sent["x-model"] == "from-model"
    assert sent["x-both"] == "explicit"
    assert sent["x-caller"] == "yes"


async def test_transform_headers_runs_last_and_is_removed_from_options(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "sk-env")
    streams = RecordingStreams()
    models = create_models()
    models.set_provider(make_provider(streams))

    async def transform(headers):
        return {**headers, "x-request-id": "r1"}

    await models.complete(
        models.get_model("acme", "m"),
        Context(messages=[user_text("hi")]),
        headers={"x-caller": "yes"},
        transform_headers=transform,
    )

    options = streams.calls[0]["options"]
    assert options["headers"]["x-request-id"] == "r1"
    assert options["headers"]["x-caller"] == "yes"
    assert "transform_headers" not in options


async def test_mixed_api_provider_dispatches_by_model_api(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "sk-env")
    completions = RecordingStreams()
    anthropic = RecordingStreams()
    models = create_models()
    models.set_provider(
        make_provider(
            None,
            models=[make_model("c", api="openai-completions"), make_model("a", api="anthropic-messages")],
            api={"openai-completions": completions, "anthropic-messages": anthropic},
        )
    )

    await models.complete(models.get_model("acme", "a"), Context(messages=[user_text("hi")]))
    assert len(anthropic.calls) == 1
    assert len(completions.calls) == 0


async def test_unconfigured_auth_yields_error_message_instead_of_raising(monkeypatch):
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    models = create_models()
    models.set_provider(make_provider(RecordingStreams()))

    result = await models.complete(models.get_model("acme", "m"), Context(messages=[user_text("hi")]))
    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert "not configured" in result.error_message


async def test_unknown_model_yields_error_message_instead_of_raising():
    models = create_models()
    orphan = make_model()
    result = await models.complete(orphan, Context(messages=[user_text("hi")]))
    assert result.stop_reason == "error"
    assert "acme" in result.error_message


class HangingStreams:
    """一个永不自行结束的适配器，用来观察取消是否传播到了这一层。"""

    def __init__(self) -> None:
        self.streams: list[AssistantMessageEventStream] = []

    def _run(self, model, context, options):
        stream = AssistantMessageEventStream()

        async def never_finish():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stream.push(
                    ErrorEvent(
                        reason="aborted",
                        error=AssistantMessage(
                            api=model.api,
                            provider=model.provider,
                            model=model.id,
                            stop_reason="aborted",
                            error_message="Request was aborted",
                        ),
                    )
                )
                raise

        stream.bind_producer(asyncio.ensure_future(never_finish()))
        self.streams.append(stream)
        return stream

    def stream(self, model, context, options):
        return self._run(model, context, options)

    def stream_simple(self, model, context, options):
        return self._run(model, context, options)


async def test_cancel_propagates_to_the_adapter_stream(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "sk-env")
    streams = HangingStreams()
    models = create_models()
    models.set_provider(make_provider(streams))

    outer = models.stream(models.get_model("acme", "m"), Context(messages=[user_text("hi")]))
    await asyncio.sleep(0.01)  # 让 _dispatch 解完认证、创建出 inner
    assert len(streams.streams) == 1

    outer.cancel()
    message = await outer.result()

    assert streams.streams[0].cancelled is True
    assert message.stop_reason == "aborted"


async def test_cancel_before_the_adapter_exists_still_propagates(monkeypatch):
    """认证解析是异步的，取消可能早于适配器 stream 被创建出来。"""
    monkeypatch.setenv("ACME_API_KEY", "sk-env")
    streams = HangingStreams()
    models = create_models()
    models.set_provider(make_provider(streams))

    outer = models.stream(models.get_model("acme", "m"), Context(messages=[user_text("hi")]))
    outer.cancel()  # inner 还没建出来

    message = await outer.result()

    assert len(streams.streams) == 1
    assert streams.streams[0].cancelled is True
    assert message.stop_reason == "aborted"


async def test_refresh_updates_dynamic_model_list():
    streams = RecordingStreams()

    async def fetch_models():
        return [make_model("dynamic")]

    provider = create_provider(
        id="acme",
        name="Acme",
        base_url="http://acme.test/v1",
        auth={"api_key": env_api_key_auth("Acme key", ["ACME_API_KEY"])},
        models=[],
        api=streams,
        fetch_models=fetch_models,
    )
    models = create_models()
    models.set_provider(provider)

    assert models.get_models("acme") == []
    await models.refresh()
    assert [m.id for m in models.get_models("acme")] == ["dynamic"]
