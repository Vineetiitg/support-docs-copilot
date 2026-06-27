import logging

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from app.core.config import settings
from app.engine.indexer import open_vector_store
from app.engine.query_transform import query_variants

logger = logging.getLogger(__name__)


def get_retriever():
    qdrant = open_vector_store()
    base_retriever = qdrant.as_retriever(search_kwargs={"k": settings.RETRIEVAL_TOP_K})

    if not settings.RERANKER_ENABLED:
        return base_retriever

    model = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=model, top_n=settings.RERANKER_TOP_N)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)


def retrieve_documents(question: str):
    retriever = get_retriever()
    documents = []
    seen = set()
    for query in query_variants(question):
        for document in retriever.invoke(query):
            key = document.metadata.get("chunk_id") or document.page_content[:120]
            if key in seen:
                continue
            seen.add(key)
            documents.append(document)
    logger.info(
        "retrieval completed query_count=%s returned_chunks=%s reranker_enabled=%s",
        len(query_variants(question)),
        len(documents),
        settings.RERANKER_ENABLED,
    )
    return documents[: settings.RETRIEVAL_TOP_K]


def get_reranked_retriever():
    return get_retriever()
