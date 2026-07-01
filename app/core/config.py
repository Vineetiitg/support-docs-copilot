from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Support Docs Copilot"
    
    # OpenRouter LLM Config
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "google/gemma-4-31b-it:free"
    
    # Qdrant Vector DB Config
    QDRANT_URL: str = ""
    QDRANT_LOCATION: str = "./qdrant_data"
    COLLECTION_NAME: str = "support_docs"
    DATA_DIR: str = "data/docs"
    
    # Embeddings Config (Lightweight ONNX cpu-only FastEmbed)
    DENSE_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # Retrieval Config
    RETRIEVAL_MODE: str = "dense"
    RETRIEVAL_TOP_K: int = 15
    RERANKER_TOP_N: int = 3
    RERANKER_ENABLED: bool = False
    MIN_RELEVANCE_SCORE: float = 0.0
    MAX_CONTEXT_CHARS: int = 12000

    # Chunking Config
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Runtime Safety
    ENABLE_GUARDRAILS: bool = True
    ENABLE_RAG_EVAL: bool = False
    MAX_QUERY_LENGTH: int = 2000
    RATE_LIMIT_PER_MINUTE: int = 30

    # Auth Config
    AUTH_ENABLED: bool = False
    SECRET_KEY: str = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # LangSmith Tracing & Observability
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "Support Docs Copilot"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
