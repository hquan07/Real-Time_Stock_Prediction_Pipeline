import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from integrations.report_artifacts import (
    SignalReportValidationError,
    build_signal_report,
    publish_signal_report,
)


class RecordingClient:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def upload_file(self, path, *, metadata, idempotency_key, content_type):
        self.calls.append((Path(path), metadata, idempotency_key, content_type))
        if self.fail:
            raise RuntimeError("CloudVault unavailable")
        return {"file_id": "signal-file-1"}


def signal_rows():
    return [
        {
            "ticker": "msft",
            "signal_type": "hold",
            "reason": "neutral",
            "confidence": Decimal("75.25"),
            "created_at": datetime(2026, 10, 14, 17, 1),
        },
        {
            "ticker": "aapl",
            "signal_type": "buy",
            "reason": "positive return",
            "confidence": Decimal("82.50"),
            "created_at": datetime(2026, 10, 14, 17, 0),
        },
    ]


def test_build_signal_report_is_compact_deterministic_and_json_safe():
    report = build_signal_report(
        signal_rows(),
        logical_date="2026-10-14",
        source_run_id="scheduled__2026-10-14",
        model_version="rf_v1",
    )

    assert report["signal_count"] == 2
    assert [row["ticker"] for row in report["signals"]] == ["AAPL", "MSFT"]
    assert report["signals"][0]["confidence"] == 82.5
    json.dumps(report)


def test_build_signal_report_rejects_duplicate_tickers():
    rows = signal_rows()
    rows[1]["ticker"] = "MSFT"
    with pytest.raises(SignalReportValidationError, match="one row per ticker"):
        build_signal_report(
            rows,
            logical_date="2026-10-14",
            source_run_id="run-1",
            model_version=None,
        )


def test_publish_signal_report_uses_run_scoped_idempotency_and_tracks_file(tmp_path):
    report = build_signal_report(
        signal_rows(),
        logical_date="2026-10-14",
        source_run_id="run-1",
        model_version="rf_v1",
    )
    client = RecordingClient()

    result = publish_signal_report(report, client=client, output_dir=tmp_path)

    assert result["status"] == "published"
    assert result["file_id"] == "signal-file-1"
    report_path, metadata, idempotency_key, content_type = client.calls[0]
    assert report_path.exists()
    assert metadata["signal_count"] == 2
    assert idempotency_key.startswith("signal-report:2026-10-14:")
    assert content_type == "application/json"
    status = json.loads(report_path.with_suffix(".cloudvault.json").read_text())
    assert status["status"] == "published"


def test_publish_failure_keeps_export_and_records_failure(tmp_path):
    report = build_signal_report(
        signal_rows(),
        logical_date="2026-10-14",
        source_run_id="run-1",
        model_version=None,
    )
    client = RecordingClient(fail=True)

    with pytest.raises(RuntimeError, match="unavailable"):
        publish_signal_report(report, client=client, output_dir=tmp_path)

    exports = [path for path in tmp_path.glob("signals_*.json") if not path.name.endswith(".cloudvault.json")]
    assert len(exports) == 1
    failure = json.loads(exports[0].with_suffix(".cloudvault.json").read_text())
    assert failure["status"] == "failed"
