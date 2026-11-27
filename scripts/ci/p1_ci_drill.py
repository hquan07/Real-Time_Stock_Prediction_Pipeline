"""Create, test and remove an isolated P1 database in CI or development."""

from __future__ import annotations

from dataclasses import replace
import os

import psycopg2
from psycopg2 import sql

from stock_pipeline.config.settings import DatabaseSettings
from stock_pipeline.db.migrations import apply_migrations

from check_p1_storage import main as check_storage


def main() -> None:
    original = DatabaseSettings.from_env()
    database = f"p1_drill_{os.getpid()}"
    admin = replace(original, database="postgres")
    connection = psycopg2.connect(**admin.connect_kwargs())
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        try:
            os.environ["POSTGRES_DB"] = database
            os.environ["P1_TEST_DATABASE"] = database
            applied = apply_migrations()
            assert len(applied) == 3, applied
            check_storage()
        finally:
            os.environ["POSTGRES_DB"] = original.database
            os.environ.pop("P1_TEST_DATABASE", None)
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                    sql.Identifier(database)))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
