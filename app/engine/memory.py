import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.core.config import settings
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
        
        # Add to user's list of sessions and global admin list
        await redis.zadd(user_sessions_key, {session_id: time.time()})
        await redis.expire(user_sessions_key, SESSION_TTL)
        await redis.zadd("all_sessions", {f"{user_id}:{session_id}": time.time()})
        await redis.expire("all_sessions", SESSION_TTL)
        logger.debug(f"Added message to session {session_id} for user {user_id}")
        
        # Trigger background summarization if chat exceeds 6 turns (sliding window)
        total_msgs = await redis.llen(key)
        if total_msgs > 6:
            asyncio.create_task(summarize_session_if_needed(user_id, session_id))
    except Exception as e:
        logger.warning(f"Failed to add session message to Redis: {e}")

async def get_session_summary(user_id: str, session_id: str) -> Optional[str]:
    try:
        redis = await get_redis_client()
        summary_key = f"session:{user_id}:{session_id}:summary"
        raw_summary = await redis.get(summary_key)
        if raw_summary:
            return raw_summary.decode("utf-8") if isinstance(raw_summary, bytes) else str(raw_summary)
        return None
    except Exception as e:
        logger.warning(f"Failed to get session summary from Redis: {e}")
        return None

async def summarize_session_if_needed(user_id: str, session_id: str) -> None:
    try:
        redis = await get_redis_client()
        key = f"session:{user_id}:{session_id}:messages"
        summary_key = f"session:{user_id}:{session_id}:summary"
        
        total_len = await redis.llen(key)
        if total_len <= 6:
            return
            
        # Get older turns (turns 1 through N-6)
        older_raw = await redis.lrange(key, 0, -7)
        if not older_raw:
            return
            
        older_msgs = []
        for raw in older_raw:
            try:
                older_msgs.append(json.loads(raw))
            except Exception:
                continue
                
        if not older_msgs:
            return
            
        old_summary = await get_session_summary(user_id, session_id)
        lines = []
        if old_summary:
            lines.append(f"Previous Summary: {old_summary}")
        for m in older_msgs:
            lines.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
            
        to_summarize = "\n".join(lines)
        
        llm = ChatOpenAI(
            model=getattr(settings, "FAST_LLM_MODEL", settings.LLM_MODEL),
            temperature=0,
            openai_api_key=getattr(settings, "FAST_LLM_API_KEY", "") or settings.OPENROUTER_API_KEY,
            openai_api_base=getattr(settings, "FAST_LLM_BASE_URL", "") or settings.OPENROUTER_BASE_URL,
            default_headers={"HTTP-Referer": "https://localhost:3000", "X-Title": "Support Docs Copilot"},
        )
        prompt = PromptTemplate(
            template="""You are a helpful technical conversation summarizer. 
Condense the following past conversation turns into a dense, 2-sentence summary capturing the main technical topics, user goals, and key details discussed.
Your output must start exactly with: "System Summary:"

Conversation to summarize:
{text_to_summarize}

Summary:""",
            input_variables=["text_to_summarize"],
        )
        chain = prompt | llm
        res = await chain.ainvoke({"text_to_summarize": to_summarize})
        summary_text = res.content.strip()
        if not summary_text.startswith("System Summary:"):
            summary_text = f"System Summary: {summary_text}"
            
        await redis.setex(summary_key, SESSION_TTL, summary_text)
        logger.info(f"Generated background conversation summary for session {session_id}")
    except Exception as e:
        logger.warning(f"Failed to summarize session history in background: {e}")

async def get_session_history(user_id: str, session_id: str, limit: int = 6) -> List[Dict[str, Any]]:
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

async def list_all_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        redis = await get_redis_client()
        entries = await redis.zrevrange("all_sessions", 0, limit - 1)
        if not entries:
            keys = await redis.keys("user_sessions:*")
            sessions = []
            for k in keys:
                uid = k.decode("utf-8").split(":")[-1] if isinstance(k, bytes) else str(k).split(":")[-1]
                user_sess = await list_user_sessions(uid, limit=10)
                sessions.extend(user_sess)
            sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
            return sessions[:limit]
            
        sessions = []
        for entry_bytes in entries:
            entry_str = entry_bytes.decode("utf-8") if isinstance(entry_bytes, bytes) else str(entry_bytes)
            if ":" in entry_str:
                uid, sid = entry_str.split(":", 1)
                meta_key = f"session:{uid}:{sid}:meta"
                raw_meta = await redis.get(meta_key)
                if raw_meta:
                    try:
                        sessions.append(json.loads(raw_meta))
                    except Exception:
                        sessions.append({"session_id": sid, "user_id": uid, "updated_at": time.time(), "last_preview": "Chat Session"})
                else:
                    sessions.append({"session_id": sid, "user_id": uid, "updated_at": time.time(), "last_preview": "Chat Session"})
        return sessions
    except Exception as e:
        logger.warning(f"Failed to list all sessions from Redis: {e}")
        return []

async def delete_session(user_id: str, session_id: str) -> bool:
    try:
        redis = await get_redis_client()
        key = f"session:{user_id}:{session_id}:messages"
        meta_key = f"session:{user_id}:{session_id}:meta"
        summary_key = f"session:{user_id}:{session_id}:summary"
        user_sessions_key = f"user_sessions:{user_id}"
        
        await redis.delete(key)
        await redis.delete(meta_key)
        await redis.delete(summary_key)
        await redis.zrem(user_sessions_key, session_id)
        await redis.zrem("all_sessions", f"{user_id}:{session_id}")
        logger.info(f"Deleted session {session_id} for user {user_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete session {session_id}: {e}")
        return False
