from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import models

from app.core.config import settings
from app.core.dependencies import get_qdrant_client


def qdrant_store_options() -> dict:
    if settings.QDRANT_URL:
        return {"url": settings.QDRANT_URL}
    return {"path": settings.QDRANT_LOCATION}


def retrieval_mode() -> RetrievalMode:
    mode = settings.RETRIEVAL_MODE.lower()
    if mode == "dense":
        return RetrievalMode.DENSE
    if mode == "sparse":
        return RetrievalMode.SPARSE
    return RetrievalMode.HYBRID


def dense_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=settings.DENSE_EMBEDDING_MODEL)


def sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=settings.SPARSE_EMBEDDING_MODEL)


def collection_exists() -> bool:
    client = get_qdrant_client()
    return any(collection.name == settings.COLLECTION_NAME for collection in client.get_collections().collections)


def open_vector_store(validate_collection_config: bool = True) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.COLLECTION_NAME,
        embedding=dense_embeddings(),
        sparse_embedding=sparse_embeddings(),
        retrieval_mode=retrieval_mode(),
        validate_collection_config=validate_collection_config,
    )


def index_documents(documents, force_recreate: bool = False) -> None:
    if force_recreate or not collection_exists():
        QdrantVectorStore.from_documents(
            documents,
            embedding=dense_embeddings(),
            sparse_embedding=sparse_embeddings(),
            collection_name=settings.COLLECTION_NAME,
            retrieval_mode=retrieval_mode(),
            force_recreate=force_recreate,
            **qdrant_store_options(),
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
