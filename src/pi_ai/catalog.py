from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from pi_ai.types import Model

DATA_DIR = Path(__file__).parent / "data"


def flatten_model_catalog(
    provider_id: str,
    groups: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Model]:
    """把按 api 分组的模型数据展开成 model id -> Model 的平表。

    数据文件里不重复存 id / provider / api，这里从分组结构补回。
    """
    models: dict[str, Model] = {}
    for api, entries in groups.items():
        for model_id, fields in entries.items():
            models[model_id] = Model.model_validate(
                {**fields, "id": model_id, "provider": provider_id, "api": api}
            )
    return models


@cache
def _read_data_file(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def load_provider_models(provider_id: str, data_dir: Path | None = None) -> dict[str, Model]:
    """从 data/<provider>.json 读取并展开模型目录。"""
    directory = DATA_DIR if data_dir is None else data_dir
    path = directory / f"{provider_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no model catalog for provider {provider_id!r} at {path}")
    groups = json.loads(_read_data_file(str(path)))
    return flatten_model_catalog(provider_id, groups)
