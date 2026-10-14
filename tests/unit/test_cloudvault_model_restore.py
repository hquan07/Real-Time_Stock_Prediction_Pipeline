import hashlib
import json
from pathlib import Path

import pytest

from integrations.model_restore import (
    ModelRestoreError,
    activate_local_version,
    restore_model_bundle,
    validate_restore_manifest,
)


def digest(body):
    return hashlib.sha256(body).hexdigest()


class DownloadClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def download_file(self, file_id, destination):
        self.calls.append(file_id)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.content[file_id])
        return target


@pytest.fixture
def remote_bundle():
    version = "random_forest_v20261014_120000"
    files = {
        "file-model": b"model",
        "file-scaler": b"scaler",
        "file-metadata": json.dumps(
            {
                "version": version,
                "model_type": "random_forest",
                "features": ["close", "volume"],
                "metrics": {"train_rmse": 0.1, "test_rmse": 0.2, "test_mae": 0.15},
                "created_at": "2026-10-14T12:00:00+00:00",
            }
        ).encode(),
    }
    manifest = {
        "schema_version": 1,
        "project_id": "real-time-stock",
        "artifact_type": "validated_model_bundle",
        "version": version,
        "model_type": "random_forest",
        "validation": {"status": "passed"},
        "files": [
            {"name": "model.pkl", "file_id": "file-model", "sha256": digest(files["file-model"]), "size_bytes": len(files["file-model"])},
            {"name": "scaler.pkl", "file_id": "file-scaler", "sha256": digest(files["file-scaler"]), "size_bytes": len(files["file-scaler"])},
            {"name": "metadata.json", "file_id": "file-metadata", "sha256": digest(files["file-metadata"]), "size_bytes": len(files["file-metadata"])},
        ],
    }
    files["manifest-file"] = json.dumps(manifest).encode()
    return version, files, manifest


def test_restore_downloads_exact_bundle_and_does_not_activate_by_default(tmp_path, remote_bundle):
    version, content, _ = remote_bundle
    client = DownloadClient(content)

    result = restore_model_bundle(
        "manifest-file",
        client=client,
        registry_dir=tmp_path,
    )

    assert result["version"] == version
    assert result["activated"] is False
    assert client.calls == ["manifest-file", "file-model", "file-scaler", "file-metadata"]
    assert (tmp_path / version / "model.pkl").read_bytes() == b"model"
    assert not (tmp_path / "latest.json").exists()


def test_restore_with_activation_retains_previous_version_for_rollback(tmp_path, remote_bundle):
    version, content, _ = remote_bundle
    previous = tmp_path / "rf_previous"
    previous.mkdir()
    (previous / "model.pkl").write_bytes(b"old-model")
    (previous / "scaler.pkl").write_bytes(b"old-scaler")
    (previous / "metadata.json").write_text(
        json.dumps(
            {
                "version": "rf_previous",
                "model_type": "rf",
                "features": ["close"],
                "metrics": {"train_rmse": 0.2, "test_rmse": 0.3, "test_mae": 0.25},
                "created_at": "2026-10-13T12:00:00+00:00",
            }
        )
    )
    (tmp_path / "latest.json").write_text(json.dumps({"latest_version": "rf_previous"}))

    result = restore_model_bundle(
        "manifest-file",
        client=DownloadClient(content),
        registry_dir=tmp_path,
        activate=True,
    )

    assert result["previous_version"] == "rf_previous"
    assert previous.exists()
    assert json.loads((tmp_path / "latest.json").read_text())["latest_version"] == version
    assert activate_local_version(tmp_path, "rf_previous") == version


def test_restore_rejects_checksum_mismatch_without_installing(tmp_path, remote_bundle):
    version, content, _ = remote_bundle
    content["file-model"] = b"tampered"

    with pytest.raises(ModelRestoreError, match="size does not match"):
        restore_model_bundle(
            "manifest-file",
            client=DownloadClient(content),
            registry_dir=tmp_path,
        )

    assert not (tmp_path / version).exists()
    assert not list(tmp_path.glob(".cloudvault-restore-*"))


def test_manifest_rejects_cross_project_and_path_traversal(remote_bundle):
    _, _, manifest = remote_bundle
    manifest["project_id"] = "another-project"
    with pytest.raises(ModelRestoreError, match="different"):
        validate_restore_manifest(manifest)

    manifest["project_id"] = "real-time-stock"
    manifest["version"] = "../escape"
    with pytest.raises(ModelRestoreError, match="version"):
        validate_restore_manifest(manifest)
