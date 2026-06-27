from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Support Docs Copilot"
    
    # Ollama LLM Config
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "llama3"
    
    # Qdrant Vector DB Config
    QDRANT_URL: str = ""
    QDRANT_LOCATION: str = "./qdrant_data"
    COLLECTION_NAME: str = "support_docs"
    
    # Embeddings Config
    DENSE_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # Retrieval Config
    RETRIEVAL_MODE: str = "hybrid"
    RETRIEVAL_TOP_K: int = 15
    RERANKER_TOP_N: int = 3
    RERANKER_ENABLED: bool = True
    MIN_RELEVANCE_SCORE: float = 0.0
    MAX_CONTEXT_CHARS: int = 12000

    # Chunking Config
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Runtime Safety
    ENABLE_GUARDRAILS: bool = True
    ENABLE_RAG_EVAL: bool = False
    MAX_QUERY_LENGTH: int = 2000

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
