"""Create and publish compact, portable stock signal reports."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from integrations.cloudvault import CloudVaultClient, CloudVaultError, DEFAULT_PROJECT_ID


class SignalReportValidationError(ValueError):
    """Raised when committed signal rows cannot form a safe report."""


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _response_file_id(response: Mapping[str, Any]) -> str:
    file_id = response.get("file_id") or response.get("id")
    if not isinstance(file_id, str) or not file_id:
        raise CloudVaultError("CloudVault upload response did not contain file_id", code="INVALID_RESPONSE")
    return file_id


def build_signal_report(
    signals: Sequence[Mapping[str, Any]],
    *,
    logical_date: str | date,
    source_run_id: str,
    model_version: str | None,
) -> dict[str, Any]:
    normalized_date = logical_date.isoformat() if isinstance(logical_date, date) else logical_date
    try:
        date.fromisoformat(normalized_date)
    except (TypeError, ValueError) as exc:
        raise SignalReportValidationError("logical_date must use YYYY-MM-DD") from exc
    if not source_run_id.strip():
        raise SignalReportValidationError("source_run_id must not be empty")
    if not signals:
        raise SignalReportValidationError("At least one committed signal is required")

    normalized_signals: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()
    created_at_values: list[str] = []
    for signal in signals:
        ticker = str(signal.get("ticker", "")).strip().upper()
        signal_type = str(signal.get("signal_type", "")).strip().upper()
        if not ticker or signal_type not in {"BUY", "SELL", "HOLD"}:
            raise SignalReportValidationError("Each signal requires a ticker and BUY/SELL/HOLD type")
        if ticker in seen_tickers:
            raise SignalReportValidationError("Signal report must contain at most one row per ticker")
        seen_tickers.add(ticker)
        created_at = _json_value(signal.get("created_at"))
        if not isinstance(created_at, str) or not created_at:
            raise SignalReportValidationError("Each signal requires created_at")
        created_at_values.append(created_at)
        normalized_signals.append(
            {
                "ticker": ticker,
                "signal_type": signal_type,
                "reason": str(signal.get("reason") or ""),
                "confidence": _json_value(signal.get("confidence")),
                "created_at": created_at,
            }
        )

    normalized_signals.sort(key=lambda item: item["ticker"])
    return {
        "schema_version": 1,
        "project_id": DEFAULT_PROJECT_ID,
        "artifact_type": "daily_signal_report",
        "logical_date": normalized_date,
        "source_run_id": source_run_id,
        "model_version": model_version,
        "generated_at": max(created_at_values),
        "signal_count": len(normalized_signals),
        "signals": normalized_signals,
    }


def publish_signal_report(
    report: Mapping[str, Any],
    *,
    client: CloudVaultClient,
    output_dir: str | Path,
    retention_class: str = "report",
) -> dict[str, Any]:
    logical_date = str(report.get("logical_date", ""))
    source_run_id = str(report.get("source_run_id", ""))
    if not logical_date or not source_run_id:
        raise SignalReportValidationError("Report is missing logical_date or source_run_id")

    run_digest = hashlib.sha256(source_run_id.encode("utf-8")).hexdigest()[:16]
    report_path = Path(output_dir) / f"signals_{logical_date}_{run_digest}.json"
    status_path = report_path.with_suffix(".cloudvault.json")
    _atomic_write_json(report_path, report)
    try:
        response = client.upload_file(
            report_path,
            metadata={
                "artifact_type": "daily_signal_report",
                "logical_date": logical_date,
                "source_run_id": source_run_id,
                "retention_class": retention_class,
                "model_version": report.get("model_version"),
                "signal_count": report.get("signal_count"),
                "object_path": f"reports/signals/{logical_date.replace('-', '/')}/{run_digest}.json",
            },
            idempotency_key=f"signal-report:{logical_date}:{run_digest}",
            content_type="application/json",
        )
        result = {
            "status": "published",
            "project_id": DEFAULT_PROJECT_ID,
            "file_id": _response_file_id(response),
            "report_path": str(report_path),
            "logical_date": logical_date,
            "source_run_id": source_run_id,
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(status_path, result)
        return result
    except Exception as exc:
        _atomic_write_json(
            status_path,
            {
                "status": "failed",
                "project_id": DEFAULT_PROJECT_ID,
                "report_path": str(report_path),
                "logical_date": logical_date,
                "source_run_id": source_run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
