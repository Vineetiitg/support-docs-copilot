import logging
from typing import List, Tuple
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_rerank_model = None
_nli_model = None


def get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading local CrossEncoder rerank model: cross-encoder/ms-marco-MiniLM-L-6-v2")
            _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        except Exception as e:
            logger.error(f"Failed to load rerank CrossEncoder model: {e}")
            raise
    return _rerank_model


def get_nli_model():
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading local CrossEncoder NLI model: cross-encoder/nli-deberta-v3-small")
            _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small", max_length=512)
        except Exception as e:
            logger.error(f"Failed to load NLI CrossEncoder model: {e}")
            raise
    return _nli_model


def rerank_documents(question: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    if not documents or len(documents) <= 1:
        return documents

    try:
        model = get_rerank_model()
        pairs = [(question, doc.page_content) for doc in documents]
        scores = model.predict(pairs)
        
        doc_with_scores = list(zip(documents, scores))
        doc_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        reranked = []
        for doc, score in doc_with_scores[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)
            
        logger.info(f"Reranked {len(documents)} docs down to top-{len(reranked)} (highest score: {doc_with_scores[0][1]:.4f})")
        return reranked
    except Exception as e:
        logger.warning(f"Reranking failed ({e}), falling back to top_k truncation without scoring.")
        return documents[:top_k]


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
