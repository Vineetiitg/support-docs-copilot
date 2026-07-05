import threading
from typing import Any

import requests
from qdrant_client import QdrantClient

from app.core.config import settings

_qdrant_client: QdrantClient | None = None
_client_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    with _client_lock:
        if _qdrant_client is None:
            if settings.QDRANT_URL:
                _qdrant_client = QdrantClient(url=settings.QDRANT_URL)
            else:
                _qdrant_client = QdrantClient(path=settings.QDRANT_LOCATION)
        return _qdrant_client


def check_openrouter() -> dict[str, Any]:
    try:
        headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"}
        url = f"{settings.OPENROUTER_BASE_URL.rstrip('/v1').rstrip('/')}/api/v1/auth/key" if "openrouter.ai" in settings.OPENROUTER_BASE_URL else f"{settings.OPENROUTER_BASE_URL}/models"
        response = requests.get(url, headers=headers, timeout=5)
        return {"ok": response.ok or response.status_code == 200, "status_code": response.status_code}
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
