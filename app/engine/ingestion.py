import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from app.core.config import settings

def ingest_documents(data_dir: str = "data/docs"):
    print(f"Loading documents from {data_dir}...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("No documents found. Please place some text documents into data/docs first.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    dense_embeddings = FastEmbedEmbeddings(model_name=settings.DENSE_EMBEDDING_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name=settings.SPARSE_EMBEDDING_MODEL)

    store_options = {
        "url": settings.QDRANT_URL,
    } if settings.QDRANT_URL else {
        "path": settings.QDRANT_LOCATION,
    }

    QdrantVectorStore.from_documents(
        chunks,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        collection_name=settings.COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
        force_recreate=True,
        **store_options,
    )
    print("Ingestion complete! Hybrid index is built.")

if __name__ == "__main__":
    ingest_documents()
