"""Environment-backed settings shared by pipeline and API runtimes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    database: str
    user: str
    password: Optional[str]

    @classmethod
    def from_env(cls, *, require_password: bool = True) -> "DatabaseSettings":
        password = _required("POSTGRES_PASSWORD") if require_password else os.getenv("POSTGRES_PASSWORD")
        return cls(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "stockdb"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=password,
        )

    def connect_kwargs(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }

    @property
    def sqlalchemy_url(self) -> str:
        if not self.password:
            raise RuntimeError("POSTGRES_PASSWORD is required to build a database URL")
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def jdbc_url(self) -> str:
        return f"jdbc:postgresql://{self.host}:{self.port}/{self.database}"


@dataclass(frozen=True)
class KafkaSettings:
    bootstrap_servers: str
    topic: str
    schema_path: Path

    @classmethod
    def from_env(cls) -> "KafkaSettings":
        return cls(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"),
            topic=os.getenv("KAFKA_TOPIC", "stock_raw_data"),
            schema_path=Path(
                os.getenv(
                    "STOCK_SCHEMA_PATH",
                    str(PROJECT_ROOT / "packages/pipeline/src/stock_pipeline/ingestion/schema/stock_schema.avsc"),
                )
            ),
        )


@dataclass(frozen=True)
class RedisSettings:
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "RedisSettings":
        return cls(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
        )


def cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
