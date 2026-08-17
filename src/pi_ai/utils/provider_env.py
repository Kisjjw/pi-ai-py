from __future__ import annotations

import os
from collections.abc import Mapping


def get_provider_env_value(name: str, env: Mapping[str, str] | None = None) -> str | None:
    """按 provider 级 env、进程环境变量的顺序解析配置值。

    空字符串按"未设置"处理，与 TS 版的 `||` 语义一致。
    TS 侧还有一段 Bun 沙箱下读 /proc/self/environ 的兜底，Python 不需要。
    """
    if env:
        scoped = env.get(name)
        if scoped:
            return scoped
    return os.environ.get(name) or None
