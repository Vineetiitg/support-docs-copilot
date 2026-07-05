import logging

from app.core.config import settings
from app.engine.indexer import open_vector_store
from app.engine.query_transform import query_variants

logger = logging.getLogger(__name__)


def get_retriever():
    qdrant = open_vector_store()
    return qdrant.as_retriever(search_kwargs={"k": settings.RETRIEVAL_TOP_K})


async def retrieve_documents(question: str, chat_history: list[dict] = None):
    qdrant = open_vector_store()
    retriever = get_retriever()
    documents = []
    seen = set()

    try:
        if hasattr(qdrant, "asimilarity_search_with_score"):
            direct_results = await qdrant.asimilarity_search_with_score(question, k=settings.RETRIEVAL_TOP_K)
        else:
            direct_results = [(doc, 0.85) for doc in await retriever.ainvoke(question)]
    except Exception:
        direct_results = [(doc, 0.85) for doc in await retriever.ainvoke(question)]

    for document, score in direct_results:
        key = document.metadata.get("chunk_id") or document.page_content[:120]
        if key not in seen:
            seen.add(key)
            document.metadata["similarity_score"] = float(score)
            documents.append(document)

    top_sim = max([d.metadata.get("similarity_score", 0.0) for d in documents] + [0.0])
    if top_sim >= 0.65 and documents:
        logger.info(f"Direct retrieval hit high similarity ({top_sim:.4f} >= 0.65). Skipping LLM query expansion!")
        return documents[: settings.RETRIEVAL_TOP_K]

    variants = await query_variants(question, chat_history)
    for query in variants:
        if query == question:
            continue
        try:
            if hasattr(qdrant, "asimilarity_search_with_score"):
                results = await qdrant.asimilarity_search_with_score(query, k=settings.RETRIEVAL_TOP_K)
            else:
                results = [(doc, 0.85) for doc in await retriever.ainvoke(query)]
        except Exception:
            results = [(doc, 0.85) for doc in await retriever.ainvoke(query)]
            
        for document, score in results:
            key = document.metadata.get("chunk_id") or document.page_content[:120]
            if key in seen:
                continue
            seen.add(key)
            document.metadata["similarity_score"] = float(score)
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
