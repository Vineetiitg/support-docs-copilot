import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_PATH = Path("data/document_registry.json")


@dataclass
class DocumentRecord:
    doc_id: str
    source: str
    source_path: str
    content_hash: str
    chunk_count: int
    ingested_at: str


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(registry: dict[str, dict], path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def hash_exists(content_hash: str, registry: dict[str, dict]) -> bool:
    return any(record["content_hash"] == content_hash for record in registry.values())


def upsert_record(
    doc_id: str,
    source: str,
    source_path: str,
    content_hash: str,
    chunk_count: int,
    registry: dict[str, dict],
) -> None:
    registry[doc_id] = asdict(
        DocumentRecord(
            doc_id=doc_id,
            source=source,
            source_path=source_path,
            content_hash=content_hash,
            chunk_count=chunk_count,
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
    )
