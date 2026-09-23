import os
import asyncpg
from loguru import logger

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5433)),
    "database": os.getenv("POSTGRES_DB", "stockdb"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "Huyquan1607")
}

async def get_db_pool():
    try:
        pool = await asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=10)
        return pool
    except Exception as e:
        logger.error(f"Failed to create asyncpg pool: {e}")
        return None
