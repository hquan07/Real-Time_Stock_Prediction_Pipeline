"""Manually restore a verified CloudVault model bundle into the local registry."""

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
from integrations.model_restore import activate_local_version, restore_model_bundle


DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "src/machine_learning/artifacts/registry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore or roll back a verified local model version")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--manifest-file-id", help="CloudVault file_id of a model bundle manifest")
    action.add_argument("--rollback-to", help="Already verified local version to activate")
    parser.add_argument("--registry-dir", default=os.getenv("MODEL_REGISTRY_DIR", str(DEFAULT_REGISTRY_DIR)))
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Atomically update latest.json after restore; omitted by default",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rollback_to:
        previous = activate_local_version(args.registry_dir, args.rollback_to)
        print(
            json.dumps(
                {
                    "status": "activated",
                    "version": args.rollback_to,
                    "previous_version": previous,
                },
                sort_keys=True,
            )
        )
        return 0

    result = restore_model_bundle(
        args.manifest_file_id,
        client=CloudVaultClient(CloudVaultConfig.from_env()),
        registry_dir=args.registry_dir,
        activate=args.activate,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
