from typing import Any

import requests
from qdrant_client import QdrantClient

from app.core.config import settings


def get_qdrant_client() -> QdrantClient:
    if settings.QDRANT_URL:
        return QdrantClient(url=settings.QDRANT_URL)
    return QdrantClient(path=settings.QDRANT_LOCATION)


def check_openrouter() -> dict[str, Any]:
    try:
        headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"}
        response = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=3)
        return {"ok": response.ok, "status_code": response.status_code}
    except Exception as exc:
        return {"ok": bool(settings.OPENROUTER_API_KEY), "error": str(exc)}


def check_qdrant() -> dict[str, Any]:
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        names = [collection.name for collection in collections.collections]
        return {
            "ok": settings.COLLECTION_NAME in names,
            "collection": settings.COLLECTION_NAME,
            "available_collections": names,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
