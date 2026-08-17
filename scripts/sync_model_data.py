"""把 TS 侧生成的模型目录 JSON 同步到 Python 包内。

用法：
    cd packages/ai && npm run hydrate-model-data
    cd ../../pi-ai-py && python scripts/sync_model_data.py

TS 侧的 src/providers/data/ 是 gitignore 的生成产物；Python 包内的副本
纳入版本控制，保证安装即可用。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REQUIRED_MODEL_FIELDS = ("name", "baseUrl", "input", "cost", "contextWindow", "maxTokens")
REQUIRED_COST_FIELDS = ("input", "output", "cacheRead", "cacheWrite")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "packages" / "ai" / "src" / "providers" / "data"
DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "src" / "pi_ai" / "data"


def validate(path: Path) -> list[str]:
    problems: list[str] = []
    groups = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(groups, dict):
        return [f"{path.name}: top level must be an object keyed by api id"]

    for api, entries in groups.items():
        if not isinstance(entries, dict):
            problems.append(f"{path.name}: group {api!r} must be an object")
            continue
        for model_id, fields in entries.items():
            for required in REQUIRED_MODEL_FIELDS:
                if required not in fields:
                    problems.append(f"{path.name}: {api}/{model_id} missing {required!r}")
            cost = fields.get("cost")
            if isinstance(cost, dict):
                for required in REQUIRED_COST_FIELDS:
                    if required not in cost:
                        problems.append(f"{path.name}: {api}/{model_id} cost missing {required!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--check", action="store_true", help="只校验，不写入")
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"source not found: {args.source}", file=sys.stderr)
        print("run 'npm run hydrate-model-data' in packages/ai first", file=sys.stderr)
        return 1

    # .manifest.json 是生成元数据（schema 版本、生成时间、结构哈希），不是厂商目录，
    # 照常拷贝以保留数据来源信息，但不参与结构校验。
    all_files = sorted(args.source.glob("*.json"))
    catalogs = [path for path in all_files if not path.name.startswith(".")]
    if not catalogs:
        print(f"no provider catalogs in {args.source}", file=sys.stderr)
        return 1

    problems: list[str] = []
    for path in catalogs:
        problems.extend(validate(path))

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    if args.check:
        print(f"validated {len(catalogs)} catalog files")
        return 0

    args.target.mkdir(parents=True, exist_ok=True)
    for path in all_files:
        shutil.copy2(path, args.target / path.name)
    print(f"synced {len(catalogs)} catalog files (+{len(all_files) - len(catalogs)} metadata) to {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
