#!/usr/bin/env python3
"""Parse repository-owned JSON and YAML configuration files."""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    json_paths = [
        ROOT / "packages/pipeline/src/stock_pipeline/ingestion/schema/stock_schema.avsc",
        ROOT / "apps/web/package.json",
    ]
    yaml_paths = [
        ROOT / "docker-compose.yml",
        ROOT / "infrastructure/compose/pilot.yml",
        ROOT / ".pre-commit-config.yaml",
        ROOT / ".github/workflows/p0-quality.yml",
    ]
    for path in json_paths:
        json.loads(path.read_text())
    for path in yaml_paths:
        yaml.safe_load(path.read_text())
    print(f"Validated {len(json_paths)} JSON and {len(yaml_paths)} YAML files.")


if __name__ == "__main__":
    main()
