import hashlib
import json
from pathlib import Path

import pytest

from integrations.model_artifacts import (
    ModelBundleValidationError,
    ModelValidationPolicy,
    publish_model_bundle,
    validate_model_bundle,
)


class RecordingClient:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def upload_file(self, path, *, metadata, idempotency_key, content_type):
        path = Path(path)
        self.calls.append((path, metadata, idempotency_key, content_type))
        if self.fail_on == path.name:
            raise RuntimeError("simulated outage")
        return {"file_id": f"file-{path.name}"}


@pytest.fixture
def model_bundle(tmp_path):
    version = "random_forest_v20261014_120000"
    version_dir = tmp_path / version
    version_dir.mkdir()
    (version_dir / "model.pkl").write_bytes(b"model")
    (version_dir / "scaler.pkl").write_bytes(b"scaler")
    (version_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": version,
                "model_type": "random_forest",
                "features": ["close", "volume"],
                "metrics": {"train_rmse": 0.1, "test_rmse": 0.2, "test_mae": 0.15},
                "created_at": "2026-10-14T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return version_dir


def test_validate_bundle_builds_checksums_without_loading_pickle(model_bundle):
    bundle = validate_model_bundle(model_bundle)

    assert bundle.version == model_bundle.name
    assert bundle.features == ("close", "volume")
    assert {item["name"] for item in bundle.files} == {
        "model.pkl",
        "scaler.pkl",
        "metadata.json",
    }
    model_file = next(item for item in bundle.files if item["name"] == "model.pkl")
    assert model_file["sha256"] == hashlib.sha256(b"model").hexdigest()


def test_validate_bundle_rejects_missing_unexpected_and_failed_quality_gate(model_bundle):
    (model_bundle / "extra.bin").write_bytes(b"not allowed")
    with pytest.raises(ModelBundleValidationError, match="unexpected"):
        validate_model_bundle(model_bundle)

    (model_bundle / "extra.bin").unlink()
    with pytest.raises(ModelBundleValidationError, match="quality gate"):
        validate_model_bundle(
            model_bundle,
            policy=ModelValidationPolicy(max_test_rmse=0.1),
        )


def test_publish_uploads_components_then_manifest_with_stable_keys(model_bundle):
    bundle = validate_model_bundle(model_bundle)
    client = RecordingClient()

    result = publish_model_bundle(
        bundle,
        client=client,
        logical_date="2026-10-14",
        source_run_id="scheduled__2026-10-14",
        ticker_scope=["AAPL", "MSFT"],
        target="target_return",
        training_window_days=90,
    )

    assert result["status"] == "published"
    assert result["manifest_file_id"] == "file-manifest.json"
    assert [call[2] for call in client.calls] == [
        f"model:{bundle.version}:model.pkl",
        f"model:{bundle.version}:scaler.pkl",
        f"model:{bundle.version}:metadata.json",
        f"model:{bundle.version}:manifest",
    ]
    manifest = json.loads((model_bundle / "manifest.json").read_text())
    assert manifest["validation"]["status"] == "passed"
    assert manifest["files"][0]["file_id"] == "file-model.pkl"
    status = json.loads((model_bundle / "cloudvault_publish.json").read_text())
    assert status["status"] == "published"


def test_publish_failure_is_recorded_without_removing_local_bundle(model_bundle):
    bundle = validate_model_bundle(model_bundle)
    client = RecordingClient(fail_on="scaler.pkl")

    with pytest.raises(RuntimeError, match="simulated outage"):
        publish_model_bundle(
            bundle,
            client=client,
            logical_date="2026-10-14",
            source_run_id="run-1",
            ticker_scope=["ALL"],
            target="target_return",
            training_window_days=90,
        )

    assert (model_bundle / "model.pkl").read_bytes() == b"model"
    status = json.loads((model_bundle / "cloudvault_publish.json").read_text())
    assert status["status"] == "failed"
    assert status["uploaded_files"][0]["file_id"] == "file-model.pkl"
