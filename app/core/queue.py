import redis.asyncio as aioredis
from arq import create_pool
from arq.connections import RedisSettings, ArqRedis
from app.core.config import settings

_redis_client: aioredis.Redis | None = None
_arq_pool: ArqRedis | None = None


def get_redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.REDIS_URL)


async def get_redis_client() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


async def get_arq_pool() -> ArqRedis:
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(get_redis_settings())
    return _arq_pool


async def close_queue_connections() -> None:
    global _redis_client, _arq_pool
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
    if _arq_pool:
        await _arq_pool.close()
        _arq_pool = None
