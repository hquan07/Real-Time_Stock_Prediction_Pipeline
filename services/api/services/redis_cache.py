import json
import redis.asyncio as redis
from loguru import logger
from stock_pipeline.config.settings import RedisSettings

settings = RedisSettings.from_env()

# Initialize Redis client
redis_client = redis.Redis(host=settings.host, port=settings.port, decode_responses=True)

async def get_cached_data(key: str):
    """Retrieve JSON data from Redis cache."""
    try:
        data = await redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(f"Redis get error for {key}: {e}")
        return None

async def set_cached_data(key: str, data: list, expire: int = 300):
    """Store JSON data in Redis cache with an expiration."""
    try:
        await redis_client.set(key, json.dumps(data), ex=expire)
    except Exception as e:
        logger.error(f"Redis set error for {key}: {e}")
