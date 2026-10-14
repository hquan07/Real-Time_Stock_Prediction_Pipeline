"""Verified CloudVault model restore without loading untrusted pickle content."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from integrations.cloudvault import CloudVaultClient, DEFAULT_PROJECT_ID
from integrations.model_artifacts import (
    REQUIRED_BUNDLE_FILES,
    ModelBundleValidationError,
    _atomic_write_json,
    validate_model_bundle,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ModelRestoreError(RuntimeError):
    """Raised when a remote model bundle cannot be safely restored."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRestoreError("Downloaded manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ModelRestoreError("Downloaded manifest must contain a JSON object")
    return manifest


def validate_restore_manifest(manifest: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if manifest.get("schema_version") != 1:
        raise ModelRestoreError("Unsupported model manifest schema version")
    if manifest.get("project_id") != DEFAULT_PROJECT_ID:
        raise ModelRestoreError("Manifest belongs to a different CloudVault project")
    if manifest.get("artifact_type") != "validated_model_bundle":
        raise ModelRestoreError("Manifest is not a validated model bundle")
    if manifest.get("validation", {}).get("status") != "passed":
        raise ModelRestoreError("Manifest does not record passed validation")

    version = manifest.get("version")
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ModelRestoreError("Manifest model version is invalid")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ModelRestoreError("Manifest files must be a list")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ModelRestoreError("Manifest file entry must be an object")
        name = item.get("name")
        file_id = item.get("file_id")
        checksum = item.get("sha256")
        size = item.get("size_bytes")
        if name not in REQUIRED_BUNDLE_FILES or name in names:
            raise ModelRestoreError("Manifest has missing, duplicate or unexpected bundle files")
        if not isinstance(file_id, str) or not file_id:
            raise ModelRestoreError(f"Manifest file {name} is missing file_id")
        if not isinstance(checksum, str) or not SHA256_PATTERN.fullmatch(checksum):
            raise ModelRestoreError(f"Manifest file {name} has an invalid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ModelRestoreError(f"Manifest file {name} has an invalid size")
        names.add(name)
        normalized.append(
            {"name": name, "file_id": file_id, "sha256": checksum, "size_bytes": size}
        )
    if names != set(REQUIRED_BUNDLE_FILES):
        raise ModelRestoreError("Manifest does not contain the exact required model files")
    return version, normalized


def _verify_file(path: Path, item: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ModelRestoreError(f"Restored file is missing: {item['name']}")
    if path.stat().st_size != item["size_bytes"]:
        raise ModelRestoreError(f"Restored file size does not match: {item['name']}")
    if _sha256(path) != item["sha256"]:
        raise ModelRestoreError(f"Restored file checksum does not match: {item['name']}")


def activate_local_version(registry_dir: str | Path, version: str) -> str | None:
    """Atomically activate a verified local version while retaining the previous one."""

    root = Path(registry_dir)
    version_dir = root / version
    try:
        validate_model_bundle(version_dir)
    except ModelBundleValidationError as exc:
        raise ModelRestoreError(f"Cannot activate unverified local version: {version}") from exc

    latest_path = root / "latest.json"
    previous_version: str | None = None
    if latest_path.is_file():
        try:
            current = json.loads(latest_path.read_text(encoding="utf-8"))
            candidate = current.get("latest_version") if isinstance(current, dict) else None
            if isinstance(candidate, str) and candidate:
                previous_version = candidate
        except json.JSONDecodeError as exc:
            raise ModelRestoreError("Existing latest.json is invalid") from exc
    _atomic_write_json(
        latest_path,
        {
            "latest_version": version,
            "previous_version": previous_version,
            "activated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return previous_version


def restore_model_bundle(
    manifest_file_id: str,
    *,
    client: CloudVaultClient,
    registry_dir: str | Path,
    activate: bool = False,
) -> dict[str, Any]:
    """Download, verify and atomically install one bundle; never unpickle it."""

    if not manifest_file_id.strip():
        raise ModelRestoreError("manifest_file_id must not be empty")
    registry_root = Path(registry_dir)
    registry_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(dir=registry_root, prefix=".cloudvault-restore-"))
    try:
        downloaded_manifest = staging_root / "manifest.download.json"
        client.download_file(manifest_file_id, downloaded_manifest)
        manifest = _read_manifest(downloaded_manifest)
        version, files = validate_restore_manifest(manifest)
        destination = registry_root / version
        staging = staging_root / version

        if destination.exists():
            if not destination.is_dir():
                raise ModelRestoreError("Restore destination exists and is not a directory")
            for item in files:
                _verify_file(destination / item["name"], item)
            restored_new = False
        else:
            staging.mkdir()
            shutil.copyfile(downloaded_manifest, staging / "manifest.json")
            for item in files:
                target = staging / item["name"]
                client.download_file(item["file_id"], target)
                _verify_file(target, item)
            validate_model_bundle(staging)
            os.replace(staging, destination)
            restored_new = True

        previous_version = activate_local_version(registry_root, version) if activate else None
        result = {
            "status": "restored",
            "project_id": DEFAULT_PROJECT_ID,
            "version": version,
            "manifest_file_id": manifest_file_id,
            "registry_path": str(destination),
            "restored_new": restored_new,
            "activated": activate,
            "previous_version": previous_version,
            "restored_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(destination / "cloudvault_restore.json", result)
        return result
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
