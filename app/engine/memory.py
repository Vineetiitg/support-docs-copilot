import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from app.core.queue import get_redis_client

logger = logging.getLogger(__name__)

SESSION_TTL = 604800  # 7 days in seconds

async def add_session_message(
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    confidence: Optional[float] = None
) -> None:
    try:
        redis = await get_redis_client()
        key = f"session:{user_id}:{session_id}:messages"
        meta_key = f"session:{user_id}:{session_id}:meta"
        user_sessions_key = f"user_sessions:{user_id}"
        
        msg = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "sources": sources or [],
            "confidence": confidence
        }
        await redis.rpush(key, json.dumps(msg))
        await redis.expire(key, SESSION_TTL)
        
        # Update session metadata
        meta = {
            "session_id": session_id,
            "user_id": user_id,
            "updated_at": time.time(),
            "last_preview": content[:60] + "..." if len(content) > 60 else content
        }
        await redis.setex(meta_key, SESSION_TTL, json.dumps(meta))
        
        # Add to user's list of sessions
        await redis.zadd(user_sessions_key, {session_id: time.time()})
        await redis.expire(user_sessions_key, SESSION_TTL)
        logger.debug(f"Added message to session {session_id} for user {user_id}")
    except Exception as e:
        logger.warning(f"Failed to add session message to Redis: {e}")

async def get_session_history(user_id: str, session_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    try:
        redis = await get_redis_client()
        key = f"session:{user_id}:{session_id}:messages"
        raw_msgs = await redis.lrange(key, -limit, -1)
        if not raw_msgs:
            return []
        messages = []
        for raw in raw_msgs:
            try:
                msg = json.loads(raw)
                messages.append(msg)
            except Exception:
                continue
        return messages
    except Exception as e:
        logger.warning(f"Failed to get session history from Redis: {e}")
        return []

async def list_user_sessions(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    try:
        redis = await get_redis_client()
        user_sessions_key = f"user_sessions:{user_id}"
        session_ids = await redis.zrevrange(user_sessions_key, 0, limit - 1)
        if not session_ids:
            return []
            
        sessions = []
        for sid_bytes in session_ids:
            sid = sid_bytes.decode("utf-8") if isinstance(sid_bytes, bytes) else str(sid_bytes)
            meta_key = f"session:{user_id}:{sid}:meta"
            raw_meta = await redis.get(meta_key)
            if raw_meta:
                try:
                    sessions.append(json.loads(raw_meta))
                except Exception:
                    sessions.append({"session_id": sid, "updated_at": time.time(), "last_preview": "Chat Session"})
            else:
                sessions.append({"session_id": sid, "updated_at": time.time(), "last_preview": "Chat Session"})
        return sessions
    except Exception as e:
        logger.warning(f"Failed to list user sessions from Redis: {e}")
        return []

async def delete_session(user_id: str, session_id: str) -> bool:
    try:
        redis = await get_redis_client()
        key = f"session:{user_id}:{session_id}:messages"
        meta_key = f"session:{user_id}:{session_id}:meta"
        user_sessions_key = f"user_sessions:{user_id}"
        
        await redis.delete(key)
        await redis.delete(meta_key)
        await redis.zrem(user_sessions_key, session_id)
        logger.info(f"Deleted session {session_id} for user {user_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete session {session_id}: {e}")
        return False
