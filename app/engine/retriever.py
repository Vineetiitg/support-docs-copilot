from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from qdrant_client import QdrantClient

from app.core.config import settings

def get_reranked_retriever():
    if settings.QDRANT_URL:
        client = QdrantClient(url=settings.QDRANT_URL)
    else:
        client = QdrantClient(path=settings.QDRANT_LOCATION)
        
    dense_embeddings = FastEmbedEmbeddings(model_name=settings.DENSE_EMBEDDING_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name=settings.SPARSE_EMBEDDING_MODEL)

    qdrant = QdrantVectorStore(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    base_retriever = qdrant.as_retriever(search_kwargs={"k": 15})
    model = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=model, top_n=3)

    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
