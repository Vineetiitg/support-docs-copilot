from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import models

from app.core.config import settings
from app.core.dependencies import get_qdrant_client


def retrieval_mode() -> RetrievalMode:
    mode = settings.RETRIEVAL_MODE.lower()
    if mode == "dense":
        return RetrievalMode.DENSE
    if mode == "sparse":
        return RetrievalMode.SPARSE
    return RetrievalMode.HYBRID


_dense_embedder = None
_sparse_embedder = None

def dense_embeddings() -> FastEmbedEmbeddings:
    global _dense_embedder
    if _dense_embedder is None:
        import os
        cache_dir = "/app/data/fastembed_cache" if os.path.exists("/app") else "./data/fastembed_cache"
        os.makedirs(cache_dir, exist_ok=True)
        _dense_embedder = FastEmbedEmbeddings(model_name=settings.DENSE_EMBEDDING_MODEL, cache_dir=cache_dir)
    return _dense_embedder


def sparse_embeddings() -> FastEmbedSparse:
    global _sparse_embedder
    if _sparse_embedder is None:
        import os
        cache_dir = "/app/data/fastembed_cache" if os.path.exists("/app") else "./data/fastembed_cache"
        os.makedirs(cache_dir, exist_ok=True)
        _sparse_embedder = FastEmbedSparse(model_name=settings.SPARSE_EMBEDDING_MODEL, cache_dir=cache_dir)
    return _sparse_embedder


def collection_exists() -> bool:
    client = get_qdrant_client()
    return any(collection.name == settings.COLLECTION_NAME for collection in client.get_collections().collections)


def open_vector_store(validate_collection_config: bool = True) -> QdrantVectorStore:
    if not collection_exists():
        from langchain_core.documents import Document
        index_documents([Document(page_content="Welcome to Support Docs Copilot knowledge base.", metadata={"doc_id": "init"})], force_recreate=True)
    mode = retrieval_mode()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.COLLECTION_NAME,
        embedding=dense_embeddings(),
        sparse_embedding=sparse_embeddings() if mode != RetrievalMode.DENSE else None,
        retrieval_mode=mode,
        validate_collection_config=validate_collection_config,
    )


def index_documents(documents, force_recreate: bool = False) -> None:
    if force_recreate or not collection_exists():
        mode = retrieval_mode()
        url_or_path_kwarg = {"url": settings.QDRANT_URL} if settings.QDRANT_URL else {"path": settings.QDRANT_LOCATION}
        QdrantVectorStore.from_documents(
            documents,
            embedding=dense_embeddings(),
            sparse_embedding=sparse_embeddings() if mode != RetrievalMode.DENSE else None,
            collection_name=settings.COLLECTION_NAME,
            retrieval_mode=mode,
            force_recreate=force_recreate,
            **url_or_path_kwarg,
        )
        return

    store = open_vector_store()
    store.add_documents(documents)


def reset_collection() -> None:
    client = get_qdrant_client()
    if collection_exists():
        client.delete_collection(settings.COLLECTION_NAME)


def delete_document(doc_id: str) -> None:
    client = get_qdrant_client()
    if not collection_exists():
        return
    client.delete(
        collection_name=settings.COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
    )
