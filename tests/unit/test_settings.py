from pathlib import Path

import pytest

from config.settings import DatabaseSettings, KafkaSettings


def test_database_settings_require_password(monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        DatabaseSettings.from_env()


def test_database_settings_read_environment(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "5544")
    monkeypatch.setenv("POSTGRES_DB", "stocks")
    monkeypatch.setenv("POSTGRES_USER", "pipeline")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-only")

    settings = DatabaseSettings.from_env()

    assert settings.host == "db.internal"
    assert settings.port == 5544
    assert settings.database == "stocks"
    assert settings.user == "pipeline"
    assert settings.password == "test-only"


def test_kafka_schema_path_is_portable(monkeypatch, tmp_path: Path):
    schema_path = tmp_path / "stock.avsc"
    monkeypatch.setenv("STOCK_SCHEMA_PATH", str(schema_path))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker:29092")

    settings = KafkaSettings.from_env()

    assert settings.bootstrap_servers == "broker:29092"
    assert settings.schema_path == schema_path
