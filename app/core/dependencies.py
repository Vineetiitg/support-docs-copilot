from typing import Any

import requests
from qdrant_client import QdrantClient

from app.core.config import settings


def get_qdrant_client() -> QdrantClient:
    if settings.QDRANT_URL:
        return QdrantClient(url=settings.QDRANT_URL)
    return QdrantClient(path=settings.QDRANT_LOCATION)


def check_ollama() -> dict[str, Any]:
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return {"ok": response.ok, "status_code": response.status_code}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


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
