"""Publish validated local model bundles to the CloudVault project catalog."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from integrations.cloudvault import CloudVaultClient, CloudVaultError, DEFAULT_PROJECT_ID


REQUIRED_BUNDLE_FILES = ("model.pkl", "scaler.pkl", "metadata.json")
GENERATED_BUNDLE_FILES = {
    "manifest.json",
    "cloudvault_publish.json",
    "cloudvault_restore.json",
}
CONTENT_TYPES = {
    "model.pkl": "application/octet-stream",
    "scaler.pkl": "application/octet-stream",
    "metadata.json": "application/json",
}


class ModelBundleValidationError(ValueError):
    """Raised when a local model version is unsafe or incomplete to publish."""


@dataclass(frozen=True)
class ModelValidationPolicy:
    """Configurable quality and size gates applied before any upload."""

    max_bundle_bytes: int = 100 * 1024 * 1024
    max_test_rmse: float | None = None
    max_test_mae: float | None = None

    @classmethod
    def from_env(cls) -> "ModelValidationPolicy":
        def optional_float(name: str) -> float | None:
            raw = os.getenv(name, "").strip()
            return float(raw) if raw else None

        return cls(
            max_bundle_bytes=int(
                os.getenv("CLOUDVAULT_MODEL_BUNDLE_MAX_BYTES", str(100 * 1024 * 1024))
            ),
            max_test_rmse=optional_float("CLOUDVAULT_MAX_TEST_RMSE"),
            max_test_mae=optional_float("CLOUDVAULT_MAX_TEST_MAE"),
        )


@dataclass(frozen=True)
class ValidatedModelBundle:
    version: str
    version_dir: Path
    model_type: str
    features: tuple[str, ...]
    metrics: Mapping[str, float]
    created_at: str
    files: tuple[Mapping[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleValidationError("metadata.json is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ModelBundleValidationError("metadata.json must contain a JSON object")
    return payload


def _validate_iso_datetime(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelBundleValidationError(f"{field} must be a non-empty ISO datetime")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ModelBundleValidationError(f"{field} must be a valid ISO datetime") from exc
    return value


def validate_model_bundle(
    version_dir: str | Path,
    *,
    policy: ModelValidationPolicy | None = None,
) -> ValidatedModelBundle:
    """Validate structure, metadata, metrics and checksums without unpickling files."""

    selected_policy = policy or ModelValidationPolicy()
    root = Path(version_dir)
    if not root.is_dir():
        raise ModelBundleValidationError(f"Model version directory does not exist: {root}")
    if selected_policy.max_bundle_bytes <= 0:
        raise ModelBundleValidationError("Model bundle size limit must be positive")

    present_files = {item.name for item in root.iterdir() if item.is_file()}
    missing = set(REQUIRED_BUNDLE_FILES) - present_files
    unexpected = present_files - set(REQUIRED_BUNDLE_FILES) - GENERATED_BUNDLE_FILES
    if missing:
        raise ModelBundleValidationError(
            f"Model bundle is missing required files: {', '.join(sorted(missing))}"
        )
    if unexpected:
        raise ModelBundleValidationError(
            f"Model bundle contains unexpected files: {', '.join(sorted(unexpected))}"
        )

    metadata = _read_metadata(root / "metadata.json")
    version = metadata.get("version")
    if version != root.name:
        raise ModelBundleValidationError("Model version must match its registry directory name")

    model_type = metadata.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        raise ModelBundleValidationError("metadata.model_type must be a non-empty string")

    features = metadata.get("features")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(item, str) or not item.strip() for item in features)
        or len(features) != len(set(features))
    ):
        raise ModelBundleValidationError(
            "metadata.features must be a non-empty ordered list of unique names"
        )

    metrics = metadata.get("metrics")
    required_metrics = {"train_rmse", "test_rmse", "test_mae"}
    if not isinstance(metrics, dict) or not required_metrics.issubset(metrics):
        raise ModelBundleValidationError(
            "metadata.metrics must include train_rmse, test_rmse and test_mae"
        )
    normalized_metrics: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelBundleValidationError(f"Metric {name} must be numeric")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise ModelBundleValidationError(f"Metric {name} must be finite and non-negative")
        normalized_metrics[str(name)] = normalized

    if (
        selected_policy.max_test_rmse is not None
        and normalized_metrics["test_rmse"] > selected_policy.max_test_rmse
    ):
        raise ModelBundleValidationError("Model failed the configured test_rmse quality gate")
    if (
        selected_policy.max_test_mae is not None
        and normalized_metrics["test_mae"] > selected_policy.max_test_mae
    ):
        raise ModelBundleValidationError("Model failed the configured test_mae quality gate")

    created_at = _validate_iso_datetime(metadata.get("created_at"), "metadata.created_at")
    files: list[Mapping[str, Any]] = []
    total_size = 0
    for name in REQUIRED_BUNDLE_FILES:
        path = root / name
        size = path.stat().st_size
        if size <= 0:
            raise ModelBundleValidationError(f"Model bundle file is empty: {name}")
        total_size += size
        files.append({"name": name, "sha256": _sha256(path), "size_bytes": size})
    if total_size > selected_policy.max_bundle_bytes:
        raise ModelBundleValidationError("Model bundle exceeds the configured size limit")

    return ValidatedModelBundle(
        version=version,
        version_dir=root,
        model_type=model_type,
        features=tuple(features),
        metrics=normalized_metrics,
        created_at=created_at,
        files=tuple(files),
    )


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _response_file_id(response: Mapping[str, Any]) -> str:
    file_id = response.get("file_id") or response.get("id")
    if not isinstance(file_id, str) or not file_id:
        raise CloudVaultError("CloudVault upload response did not contain file_id", code="INVALID_RESPONSE")
    return file_id


def publish_model_bundle(
    bundle: ValidatedModelBundle,
    *,
    client: CloudVaultClient,
    logical_date: str | date,
    source_run_id: str,
    ticker_scope: Sequence[str],
    target: str,
    training_window_days: int,
    retention_class: str = "model",
) -> dict[str, Any]:
    """Upload all bundle members and a final catalog manifest with stable keys."""

    normalized_date = logical_date.isoformat() if isinstance(logical_date, date) else logical_date
    try:
        date.fromisoformat(normalized_date)
    except (TypeError, ValueError) as exc:
        raise ModelBundleValidationError("logical_date must use YYYY-MM-DD") from exc
    if not source_run_id.strip():
        raise ModelBundleValidationError("source_run_id must not be empty")
    tickers = [ticker.strip().upper() for ticker in ticker_scope if ticker.strip()]
    if not tickers:
        raise ModelBundleValidationError("ticker_scope must contain at least one ticker or ALL")
    if training_window_days <= 0:
        raise ModelBundleValidationError("training_window_days must be positive")

    status_path = bundle.version_dir / "cloudvault_publish.json"
    uploaded_files: list[dict[str, Any]] = []
    common_metadata = {
        "logical_date": normalized_date,
        "source_run_id": source_run_id,
        "retention_class": retention_class,
        "bundle_version": bundle.version,
        "model_type": bundle.model_type,
        "ticker_scope": tickers,
    }
    try:
        for file_info in bundle.files:
            name = str(file_info["name"])
            response = client.upload_file(
                bundle.version_dir / name,
                metadata={
                    **common_metadata,
                    "artifact_type": "validated_model_component",
                    "component_name": name,
                    "object_path": f"models/{bundle.model_type}/{bundle.version}/{name}",
                },
                idempotency_key=f"model:{bundle.version}:{name}",
                content_type=CONTENT_TYPES[name],
            )
            uploaded_files.append(
                {
                    **dict(file_info),
                    "file_id": _response_file_id(response),
                    "content_type": CONTENT_TYPES[name],
                }
            )

        manifest = {
            "schema_version": 1,
            "project_id": DEFAULT_PROJECT_ID,
            "artifact_type": "validated_model_bundle",
            "version": bundle.version,
            "model_type": bundle.model_type,
            "ticker_scope": tickers,
            "features": list(bundle.features),
            "target": target,
            "training_window_days": training_window_days,
            "metrics": dict(bundle.metrics),
            "python_version": platform.python_version(),
            "dependencies": _dependency_versions(),
            "serialization": "pickle",
            "created_at": bundle.created_at,
            "validation": {
                "status": "passed",
                "checks": [
                    "required_files",
                    "allowed_files",
                    "metadata_schema",
                    "finite_metrics",
                    "quality_thresholds",
                    "bundle_size",
                    "sha256",
                ],
            },
            "files": uploaded_files,
        }
        manifest_path = bundle.version_dir / "manifest.json"
        _atomic_write_json(manifest_path, manifest)
        manifest_response = client.upload_file(
            manifest_path,
            metadata={
                **common_metadata,
                "artifact_type": "validated_model_manifest",
                "object_path": f"models/{bundle.model_type}/{bundle.version}/manifest.json",
            },
            idempotency_key=f"model:{bundle.version}:manifest",
            content_type="application/json",
        )
        result = {
            "status": "published",
            "project_id": DEFAULT_PROJECT_ID,
            "version": bundle.version,
            "manifest_file_id": _response_file_id(manifest_response),
            "files": uploaded_files,
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
                "version": bundle.version,
                "uploaded_files": uploaded_files,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


def latest_registry_version(registry_dir: str | Path) -> str:
    latest_path = Path(registry_dir) / "latest.json"
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleValidationError("Registry latest.json is missing or invalid") from exc
    version = payload.get("latest_version") if isinstance(payload, dict) else None
    if not isinstance(version, str) or not version:
        raise ModelBundleValidationError("Registry latest.json does not contain latest_version")
    return version
