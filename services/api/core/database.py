import asyncpg
from loguru import logger
from stock_pipeline.config.settings import DatabaseSettings

async def get_db_pool():
    settings = DatabaseSettings.from_env()
    try:
        pool = await asyncpg.create_pool(
            **settings.connect_kwargs(), min_size=1, max_size=10
        )
        return pool
    except Exception as e:
        logger.error(f"Failed to create asyncpg pool: {e}")
        raise
