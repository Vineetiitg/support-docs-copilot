import logging

from app.core.config import settings
from app.engine.indexer import open_vector_store
from app.engine.query_transform import query_variants

logger = logging.getLogger(__name__)


def get_retriever():
    qdrant = open_vector_store()
    return qdrant.as_retriever(search_kwargs={"k": settings.RETRIEVAL_TOP_K})


async def retrieve_documents(question: str, chat_history: list[dict] = None):
    retriever = get_retriever()
    documents = []
    seen = set()
    variants = await query_variants(question, chat_history)
    for query in variants:
        for document in await retriever.ainvoke(query):
            key = document.metadata.get("chunk_id") or document.page_content[:120]
            if key in seen:
                continue
            seen.add(key)
            documents.append(document)
    logger.info(
        "retrieval completed query_count=%s returned_chunks=%s reranker_enabled=%s",
        len(variants),
        len(documents),
        settings.RERANKER_ENABLED,
    )
    return documents[: settings.RETRIEVAL_TOP_K]


def get_reranked_retriever():
    return get_retriever()
