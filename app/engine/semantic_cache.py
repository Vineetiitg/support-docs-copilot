import json
import logging
import math
from typing import Any, Dict, List, Optional
from app.core.queue import get_redis_client
from app.engine.indexer import dense_embeddings

logger = logging.getLogger(__name__)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)

async def get_cached_answer(query: str, similarity_threshold: float = 0.92) -> Optional[Dict[str, Any]]:
    try:
        redis = await get_redis_client()
        keys = await redis.keys("semantic_cache:*")
        if not keys:
            return None
            
        embedder = dense_embeddings()
        query_vec = embedder.embed_query(query)
        
        best_sim = 0.0
        best_match = None
        
        for k in keys[:100]:  # Check up to 100 recent cached items
            raw = await redis.get(k)
            if not raw:
                continue
            data = json.loads(raw)
            cached_vec = data.get("embedding")
            if not cached_vec:
                continue
                
            sim = cosine_similarity(query_vec, cached_vec)
            if sim > best_sim and sim >= similarity_threshold:
                best_sim = sim
                best_match = data
                
        if best_match:
            logger.info(f"Semantic cache HIT (similarity: {best_sim:.4f} >= {similarity_threshold}) for query: '{query}'")
            return {
                "answer": best_match["answer"],
                "sources": best_match.get("sources", []),
                "confidence": best_match.get("confidence", 0.99),
                "cached": True,
                "similarity": best_sim
            }
        logger.info(f"Semantic cache MISS (highest similarity: {best_sim:.4f} < {similarity_threshold}) for query: '{query}'")
        return None
    except Exception as e:
        logger.warning(f"Semantic cache lookup failed ({e})")
        return None

async def set_cached_answer(query: str, answer: str, sources: List[Any], confidence: float, ttl_seconds: int = 86400) -> None:
    try:
        redis = await get_redis_client()
        embedder = dense_embeddings()
        query_vec = embedder.embed_query(query)
        
        formatted_sources = []
        for s in sources:
            if isinstance(s, dict):
                formatted_sources.append(s)
            elif hasattr(s, "dict"):
                formatted_sources.append(s.dict())
            elif hasattr(s, "model_dump"):
                formatted_sources.append(s.model_dump())
            else:
                formatted_sources.append(str(s))
                
        key = f"semantic_cache:{abs(hash(query))}"
        payload = {
            "query": query,
            "embedding": query_vec,
            "answer": answer,
            "sources": formatted_sources,
            "confidence": confidence
        }
        await redis.setex(key, ttl_seconds, json.dumps(payload))
        logger.info(f"Cached semantic answer for query: '{query}'")
    except Exception as e:
        logger.warning(f"Failed to set semantic cache ({e})")
