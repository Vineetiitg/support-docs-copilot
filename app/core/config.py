import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Support Docs Copilot"
    
    # Ollama LLM Config
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    
    # Qdrant Vector DB Config
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_LOCATION: str = os.getenv("QDRANT_LOCATION", "./qdrant_data")
    COLLECTION_NAME: str = "support_docs"
    
    # Embeddings Config
    DENSE_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

settings = Settings()
