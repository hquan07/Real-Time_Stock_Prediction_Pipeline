"""CLI entrypoint for publishing the latest validated model to CloudVault."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.cloudvault import CloudVaultClient, CloudVaultConfig
from integrations.model_artifacts import (
    ModelValidationPolicy,
    latest_registry_version,
    publish_model_bundle,
    validate_model_bundle,
)


DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "src/machine_learning/artifacts/registry"


def _enabled() -> bool:
    return os.getenv("CLOUDVAULT_PUBLISH_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a validated model bundle to CloudVault")
    parser.add_argument("--registry-dir", default=os.getenv("MODEL_REGISTRY_DIR", str(DEFAULT_REGISTRY_DIR)))
    parser.add_argument("--version", help="Immutable model version; defaults to registry latest.json")
    parser.add_argument("--logical-date", required=True, help="Airflow logical date in YYYY-MM-DD format")
    parser.add_argument("--source-run-id", required=True, help="Stable Airflow DAG run identifier")
    parser.add_argument("--ticker-scope", default=os.getenv("CLOUDVAULT_MODEL_TICKERS", "ALL"))
    parser.add_argument("--target", default="target_return")
    parser.add_argument("--training-window-days", type=int, default=90)
    parser.add_argument("--retention-class", default=os.getenv("CLOUDVAULT_MODEL_RETENTION_CLASS", "model"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _enabled():
        print(json.dumps({"status": "skipped", "reason": "CLOUDVAULT_PUBLISH_ENABLED is false"}))
        return 0

    registry_dir = Path(args.registry_dir)
    version = args.version or latest_registry_version(registry_dir)
    bundle = validate_model_bundle(
        registry_dir / version,
        policy=ModelValidationPolicy.from_env(),
    )
    result = publish_model_bundle(
        bundle,
        client=CloudVaultClient(CloudVaultConfig.from_env()),
        logical_date=args.logical_date,
        source_run_id=args.source_run_id,
        ticker_scope=args.ticker_scope.split(","),
        target=args.target,
        training_window_days=args.training_window_days,
        retention_class=args.retention_class,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
