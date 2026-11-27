"""Minimal checksum-enforced SQL migration runner."""

from hashlib import sha256
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

from stock_pipeline.config.settings import DatabaseSettings, PROJECT_ROOT


MIGRATIONS_DIR = PROJECT_ROOT / "infrastructure" / "migrations"
LOCK_ID = 802_024_001


def _migration_files() -> list[Path]:
    return sorted(
        path
        for path in MIGRATIONS_DIR.glob("[0-9][0-9][0-9][0-9]_*.sql")
        if not path.name.endswith(".down.sql")
    )


def _checksum(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def apply_migrations() -> list[str]:
    settings = DatabaseSettings.from_env()
    applied_now: list[str] = []
    with psycopg2.connect(**settings.connect_kwargs()) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS public.schema_migrations (
                    version varchar(255) PRIMARY KEY,
                    checksum varchar(64) NOT NULL,
                    applied_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("SELECT version, checksum FROM public.schema_migrations")
            applied = {row["version"]: row["checksum"] for row in cursor.fetchall()}

            for path in _migration_files():
                checksum = _checksum(path)
                if path.name in applied:
                    if applied[path.name] != checksum:
                        raise RuntimeError(f"Migration checksum changed: {path.name}")
                    continue
                cursor.execute(path.read_text())
                cursor.execute(
                    "INSERT INTO public.schema_migrations (version, checksum) VALUES (%s, %s)",
                    (path.name, checksum),
                )
                applied_now.append(path.name)
    return applied_now


def rollback_latest() -> str:
    settings = DatabaseSettings.from_env()
    with psycopg2.connect(**settings.connect_kwargs()) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
            cursor.execute(
                """
                SELECT version FROM public.schema_migrations
                ORDER BY applied_at DESC, version DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            if not row:
                raise RuntimeError("No applied migration to roll back")
            version = row[0]
            down_path = MIGRATIONS_DIR / version.replace(".sql", ".down.sql")
            if not down_path.exists():
                raise RuntimeError(f"Migration has no rollback file: {version}")
            cursor.execute(down_path.read_text())
            cursor.execute(
                "DELETE FROM public.schema_migrations WHERE version = %s", (version,)
            )
    return version
