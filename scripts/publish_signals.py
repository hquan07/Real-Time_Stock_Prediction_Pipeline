"""Export committed daily signals and publish the compact report to CloudVault."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.cloudvault import CloudVaultClient, CloudVaultConfig
from integrations.model_artifacts import ModelBundleValidationError, latest_registry_version
from integrations.report_artifacts import build_signal_report, publish_signal_report


DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "src/machine_learning/artifacts/registry"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "src/machine_learning/artifacts/reports"


def _enabled() -> bool:
    configured = os.getenv(
        "CLOUDVAULT_REPORT_PUBLISH_ENABLED",
        os.getenv("CLOUDVAULT_PUBLISH_ENABLED", "false"),
    )
    return configured.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish committed daily stock signals")
    parser.add_argument("--logical-date", required=True, help="Airflow logical date in YYYY-MM-DD format")
    parser.add_argument("--source-run-id", required=True, help="Stable Airflow DAG run identifier")
    parser.add_argument("--output-dir", default=os.getenv("CLOUDVAULT_REPORT_EXPORT_DIR", str(DEFAULT_REPORT_DIR)))
    parser.add_argument("--registry-dir", default=os.getenv("MODEL_REGISTRY_DIR", str(DEFAULT_REGISTRY_DIR)))
    parser.add_argument("--retention-class", default=os.getenv("CLOUDVAULT_REPORT_RETENTION_CLASS", "report"))
    return parser.parse_args()


def _load_committed_signals(logical_date: str) -> list[dict]:
    from sqlalchemy import text

    from src.database.db_connection import get_engine

    start = date.fromisoformat(logical_date)
    end = start + timedelta(days=1)
    query = text(
        """
        SELECT DISTINCT ON (ticker)
            ticker, signal_type, reason, confidence, created_at
        FROM public.signals
        WHERE created_at >= :start_at AND created_at < :end_at
        ORDER BY ticker, created_at DESC, id DESC
        """
    )
    with get_engine().connect() as connection:
        rows = connection.execute(query, {"start_at": start, "end_at": end})
        return [dict(row._mapping) for row in rows]


def _latest_model_version(registry_dir: str) -> str | None:
    try:
        return latest_registry_version(registry_dir)
    except ModelBundleValidationError:
        return None


def main() -> int:
    args = parse_args()
    if not _enabled():
        print(json.dumps({"status": "skipped", "reason": "CloudVault report publishing is disabled"}))
        return 0

    signals = _load_committed_signals(args.logical_date)
    if not signals:
        print(json.dumps({"status": "skipped", "reason": "No committed signals for logical date"}))
        return 0

    report = build_signal_report(
        signals,
        logical_date=args.logical_date,
        source_run_id=args.source_run_id,
        model_version=_latest_model_version(args.registry_dir),
    )
    result = publish_signal_report(
        report,
        client=CloudVaultClient(CloudVaultConfig.from_env()),
        output_dir=args.output_dir,
        retention_class=args.retention_class,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
