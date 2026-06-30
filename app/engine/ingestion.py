import argparse
import uuid
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from app.core.logging import logger

from app.engine.chunking import chunk_documents
from app.engine.document_registry import (
    file_hash,
    hash_exists,
    load_registry,
    save_registry,
    upsert_record,
)
from app.engine.indexer import delete_document, index_documents, reset_collection
from app.engine.loaders import SUPPORTED_EXTENSIONS, load_document
from app.guardrails.document import filter_malicious_documents


def ingest_documents(data_dir: str = "data/docs", force: bool = False) -> None:
    logger.info(f"Loading documents from {data_dir}...")
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    registry = {} if force else load_registry()
    source_documents: list[Document] = []
    doc_id_by_source: dict[str, str] = {}

    if force:
        reset_collection()

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content_hash = file_hash(path)
        if not force and hash_exists(content_hash, registry):
            logger.info(f"Skipping unchanged document: {path.name}")
            continue

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path.name}:{content_hash}"))
        loaded = load_document(path)
        loaded, flagged = filter_malicious_documents(loaded)
        for flagged_source in flagged:
            logger.info(f"Skipping document with suspicious instructions: {flagged_source}")
        if not loaded:
            continue
        for document in loaded:
            document.metadata["doc_id"] = doc_id
            document.metadata["content_hash"] = content_hash
        source_documents.extend(loaded)
        doc_id_by_source[str(path)] = doc_id

    if not source_documents:
        logger.info("No new documents found.")
        return

    chunks = chunk_documents(source_documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")

    index_documents(chunks, force_recreate=force)

    chunk_counts = Counter(chunk.metadata["doc_id"] for chunk in chunks)
    for source_path, doc_id in doc_id_by_source.items():
        path = Path(source_path)
        upsert_record(
            doc_id=doc_id,
            source=path.name,
            source_path=str(path),
            content_hash=file_hash(path),
            chunk_count=chunk_counts[doc_id],
            registry=registry,
        )
    save_registry(registry)
    logger.info("Ingestion complete. Hybrid index is built.")


def list_documents() -> None:
    registry = load_registry()
    if not registry:
        logger.info("No indexed documents found.")
        return
    for record in registry.values():
        logger.info(f"{record['doc_id']} | {record['source']} | chunks={record['chunk_count']}")


def delete_indexed_document(doc_id: str) -> None:
    registry = load_registry()
    if doc_id not in registry:
        logger.info(f"Document not found: {doc_id}")
        return
    delete_document(doc_id)
    del registry[doc_id]
    save_registry(registry)
    logger.info(f"Deleted document: {doc_id}")


def reset_index() -> None:
    reset_collection()
    save_registry({})
    logger.info("Vector collection and document registry reset.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage support document ingestion.")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into Qdrant.")
    ingest_parser.add_argument("--data-dir", default="data/docs")
    ingest_parser.add_argument("--force", action="store_true", help="Recreate the collection before ingesting.")

    subparsers.add_parser("list", help="List indexed documents.")

    delete_parser = subparsers.add_parser("delete", help="Delete one indexed document.")
    delete_parser.add_argument("--doc-id", required=True)

    subparsers.add_parser("reset", help="Delete the vector collection and registry.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {None, "ingest"}:
        ingest_documents(data_dir=getattr(args, "data_dir", "data/docs"), force=getattr(args, "force", False))
    elif args.command == "list":
        list_documents()
    elif args.command == "delete":
        delete_indexed_document(args.doc_id)
    elif args.command == "reset":
        reset_index()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
