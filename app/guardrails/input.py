import re
import time
from collections import defaultdict, deque

from app.core.config import settings
from app.core.errors import CopilotError


_requests_by_client: dict[str, deque[float]] = defaultdict(deque)


PROMPT_INJECTION_PATTERNS = [
    "ignore previous",
    "ignore the instructions",
    "forget all previous",
    "system prompt",
    "developer message",
    "bypass system",
    "disregard instructions",
    "reveal hidden",
    "you are now an arbitrary",
    "dan",
    "do anything now",
    "ignore all constraints"
]


def validate_query(query: str) -> None:
    if not query.strip():
        raise CopilotError("Query cannot be empty.", status_code=400)
    if len(query) > settings.MAX_QUERY_LENGTH:
        raise CopilotError("Query is too long.", status_code=400)
    normalized = query.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern in normalized:
            raise CopilotError("Prompt injection attempt detected.", status_code=400)


def enforce_rate_limit(client_id: str) -> None:
    if settings.RATE_LIMIT_PER_MINUTE <= 0:
        return
    now = time.time()
    bucket = _requests_by_client[client_id]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= settings.RATE_LIMIT_PER_MINUTE:
        raise CopilotError("Rate limit exceeded.", status_code=429)
    bucket.append(now)


async def async_enforce_rate_limit(client_id: str) -> None:
    if settings.RATE_LIMIT_PER_MINUTE <= 0:
        return
    try:
        from app.core.queue import get_redis_client
        redis = await get_redis_client()
        key = f"rate_limit:{client_id}"
        current = await redis.incr(key)
        if current == 1:
            await redis.expire(key, 60)
        if current > settings.RATE_LIMIT_PER_MINUTE:
            raise CopilotError("Rate limit exceeded.", status_code=429)
    except CopilotError:
        raise
    except Exception:
        enforce_rate_limit(client_id)



def contains_pii(text: str) -> bool:
    email = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    phone = r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b"
    return bool(re.search(email, text) or re.search(phone, text))
