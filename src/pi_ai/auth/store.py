from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter

from pi_ai.auth.types import Credential, CredentialInfo, ModifyCallback

_CREDENTIAL_ADAPTER: TypeAdapter[Credential] = TypeAdapter(Credential)


class InMemoryCredentialStore:
    """进程内凭据存储，默认实现。

    modify 是唯一写入路径，按 provider 串行，保证并发刷新不会互相覆盖。
    """

    def __init__(self) -> None:
        self._entries: dict[str, Credential] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def read(self, provider_id: str) -> Credential | None:
        return self._entries.get(provider_id)

    async def list(self) -> list[CredentialInfo]:
        return [
            CredentialInfo(provider_id=provider_id, type=credential.type)
            for provider_id, credential in self._entries.items()
        ]

    async def modify(self, provider_id: str, fn: ModifyCallback) -> Credential | None:
        async with self._locks[provider_id]:
            current = self._entries.get(provider_id)
            updated = await fn(current)
            if updated is not None:
                self._entries[provider_id] = updated
            return self._entries.get(provider_id)

    async def delete(self, provider_id: str) -> None:
        async with self._locks[provider_id]:
            self._entries.pop(provider_id, None)


class FileCredentialStore:
    """文件持久化凭据存储，格式与 pi 的 auth.json 一致。"""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _load(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _save(self, entries: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    async def read(self, provider_id: str) -> Credential | None:
        raw = self._load().get(provider_id)
        if raw is None:
            return None
        return cast(Credential, _CREDENTIAL_ADAPTER.validate_python(raw))

    async def list(self) -> list[CredentialInfo]:
        return [
            CredentialInfo(provider_id=provider_id, type=raw.get("type", "api_key"))
            for provider_id, raw in self._load().items()
            if isinstance(raw, dict)
        ]

    async def modify(self, provider_id: str, fn: ModifyCallback) -> Credential | None:
        async with self._locks[provider_id]:
            entries = self._load()
            raw = entries.get(provider_id)
            current = _CREDENTIAL_ADAPTER.validate_python(raw) if raw is not None else None
            updated = await fn(current)
            if updated is not None:
                entries[provider_id] = updated.model_dump(by_alias=True, exclude_none=True)
                self._save(entries)
                return updated
            return current

    async def delete(self, provider_id: str) -> None:
        async with self._locks[provider_id]:
            entries = self._load()
            if provider_id in entries:
                del entries[provider_id]
                self._save(entries)
