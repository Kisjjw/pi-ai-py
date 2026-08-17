from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pi_ai.auth.types import ApiKeyCredential, AuthContext, AuthResult, ModelAuth


class DefaultAuthContext:
    """默认环境访问实现。

    provider 级 env 优先于进程环境变量，对应 TS 侧 StreamOptions.env 的语义。
    """

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._scoped = dict(env or {})

    async def env(self, name: str) -> str | None:
        scoped = self._scoped.get(name)
        if scoped:
            return scoped
        return os.environ.get(name) or None

    async def file_exists(self, path: str) -> bool:
        return Path(path).expanduser().is_file()


@dataclass
class EnvApiKeyAuth:
    """标准 API key 解析：已存凭据优先，其次按顺序取第一个有值的环境变量。"""

    name: str
    env_vars: Sequence[str]

    async def resolve(
        self,
        *,
        ctx: AuthContext,
        credential: ApiKeyCredential | None,
    ) -> AuthResult | None:
        if credential is not None and credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=credential.key),
                env=credential.env,
                source="stored credential",
            )
        for env_var in self.env_vars:
            value = await ctx.env(env_var)
            if value:
                return AuthResult(auth=ModelAuth(api_key=value), source=env_var)
        return None


def env_api_key_auth(name: str, env_vars: Sequence[str]) -> EnvApiKeyAuth:
    return EnvApiKeyAuth(name=name, env_vars=list(env_vars))
