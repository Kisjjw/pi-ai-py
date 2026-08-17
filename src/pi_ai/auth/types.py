from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ModelAuth(BaseModel):
    """单次请求的认证数据。表达不了成 api_key/headers/base_url 的就不是认证，是配置。"""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    api_key: str | None = Field(default=None, alias="apiKey")
    headers: dict[str, str | None] | None = None
    base_url: str | None = Field(default=None, alias="baseUrl")


class ApiKeyCredential(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    type: Literal["api_key"] = "api_key"
    key: str | None = None
    env: dict[str, str] | None = None


class OAuthCredential(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    type: Literal["oauth"] = "oauth"
    access: str
    refresh: str
    expires: int


Credential = Annotated[ApiKeyCredential | OAuthCredential, Field(discriminator="type")]


class CredentialInfo(BaseModel):
    """凭据枚举用的非敏感元数据。纯本地结构，不参与线上协议，因此不设 alias。"""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    provider_id: str
    type: Literal["api_key", "oauth"]


class AuthResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    auth: ModelAuth
    env: dict[str, str] | None = None
    source: str | None = None


class AuthContext(Protocol):
    async def env(self, name: str) -> str | None: ...

    async def file_exists(self, path: str) -> bool: ...


class ApiKeyAuth(Protocol):
    name: str

    async def resolve(
        self,
        *,
        ctx: AuthContext,
        credential: ApiKeyCredential | None,
    ) -> AuthResult | None: ...


ModifyCallback = Callable[["Credential | None"], Awaitable["Credential | None"]]


class CredentialStore(Protocol):
    async def read(self, provider_id: str) -> Credential | None: ...

    async def list(self) -> Sequence[CredentialInfo]: ...

    async def modify(self, provider_id: str, fn: ModifyCallback) -> Credential | None: ...

    async def delete(self, provider_id: str) -> None: ...
