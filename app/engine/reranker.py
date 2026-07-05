import logging
from typing import List, Tuple
from langchain_core.documents import Document
from app.core.config import settings

logger = logging.getLogger(__name__)

_cohere_client = None
_flashrank_client = None
_nli_model = None


def get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        if not settings.COHERE_API_KEY or "your_cohere" in settings.COHERE_API_KEY:
            logger.warning("COHERE_API_KEY not set or default! Cohere reranking disabled.")
            return None
        try:
            import cohere
            logger.info(f"Initializing Cohere API Client with model: {settings.RERANKER_MODEL}")
            _cohere_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            return None
    return _cohere_client


def get_flashrank_client():
    global _flashrank_client
    if _flashrank_client is None:
        try:
            from flashrank import Ranker
            import os
            cache_dir = os.environ.get("FLASHRANK_CACHE_DIR", "/app/data/flashrank_cache" if os.path.exists("/app") else "./data/flashrank_cache")
            os.makedirs(cache_dir, exist_ok=True)
            model_name = getattr(settings, "FLASHRANK_MODEL", "ms-marco-TinyBERT-L-2-v2")
            logger.info(f"Initializing local FlashRank ONNX Client with model: {model_name}")
            _flashrank_client = Ranker(model_name=model_name, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to initialize FlashRank client: {e}")
            return None
    return _flashrank_client


def rerank_with_flashrank(question: str, documents: List[Document], top_k: int) -> List[Document]:
    client = get_flashrank_client()
    if not client:
        logger.warning("FlashRank unavailable, returning un-reranked documents.")
        return documents[:top_k]
    try:
        from flashrank import RerankRequest
        passages = [
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]
        request = RerankRequest(query=question, passages=passages)
        results = client.rerank(request)[:min(top_k, len(documents))]
        
        reranked = []
        for r in results:
            idx = int(r["id"])
            doc = documents[idx]
            doc.metadata["rerank_score"] = float(r["score"])
            doc.metadata["relevance_score"] = float(r["score"])
            reranked.append(doc)
            
        highest_score = reranked[0].metadata["rerank_score"] if reranked else 0.0
        logger.info(f"FlashRank ONNX Reranked {len(documents)} docs down to top-{len(reranked)} (highest score: {highest_score:.4f})")
        return reranked
    except Exception as e:
        logger.error(f"FlashRank reranking failed ({e}), falling back to top_k truncation without scoring.")
        return documents[:top_k]


def rerank_documents(question: str, documents: List[Document], top_k: int = 3) -> List[Document]:
    if not documents or len(documents) <= 1:
        return documents

    provider = getattr(settings, "RERANKER_PROVIDER", "auto").lower()
    has_cohere_key = bool(settings.COHERE_API_KEY and settings.COHERE_API_KEY.strip() != "" and "your_cohere" not in settings.COHERE_API_KEY)

    # 1. Try Cohere API if explicitly selected or if 'auto' with a valid API key
    if provider == "cohere" or (provider == "auto" and has_cohere_key):
        client = get_cohere_client()
        if client:
            try:
                doc_texts = [doc.page_content for doc in documents]
                response = client.rerank(
                    model=settings.RERANKER_MODEL,
                    query=question,
                    documents=doc_texts,
                    top_n=min(top_k, len(documents))
                )
                
                reranked = []
                for r in response.results:
                    doc = documents[r.index]
                    doc.metadata["rerank_score"] = float(r.relevance_score)
                    doc.metadata["relevance_score"] = float(r.relevance_score)
                    reranked.append(doc)
                    
                highest_score = reranked[0].metadata["rerank_score"] if reranked else 0.0
                logger.info(f"Cohere API Reranked {len(documents)} docs down to top-{len(reranked)} (highest score: {highest_score:.4f})")
                return reranked
            except Exception as e:
                logger.warning(f"Cohere API reranking failed ({e}). Attempting seamless fallback to local FlashRank...")

    # 2. Use local FlashRank ONNX reranker (if provider=='flashrank', no Cohere key, or Cohere API fallback)
    logger.info("Using local FlashRank CPU ONNX reranker.")
    return rerank_with_flashrank(question, documents, top_k)


def evaluate_nli_groundedness(premise: str, hypothesis: str) -> Tuple[str, float]:
    try:
        model = get_nli_model()
        scores = model.predict([(premise, hypothesis)], apply_softmax=True)[0]
        id2label = getattr(model.model.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
        
        entailment_score = 0.0
        contradiction_score = 0.0
        for idx, prob in enumerate(scores):
            label = str(id2label.get(idx, "")).lower()
            if "entail" in label:
                entailment_score = float(prob)
            elif "contradict" in label:
                contradiction_score = float(prob)
                
        logger.info(f"NLI Groundedness evaluation - Entailment: {entailment_score:.4f}, Contradiction: {contradiction_score:.4f}")
        if contradiction_score < 0.40:
            return "yes", float(1.0 - contradiction_score)
        return "no", float(1.0 - contradiction_score) if contradiction_score > 0 else 0.5
    except Exception as e:
        logger.warning(f"NLI evaluation failed ({e}), defaulting to grounded=yes.")
        return "yes", 0.5
