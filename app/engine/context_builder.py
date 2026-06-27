from langchain_core.documents import Document

from app.core.config import settings


def build_context(documents: list[Document]) -> str:
    parts: list[str] = []
    total_chars = 0
    for document in dedupe_documents(documents):
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        label = f"{source}, page {page}" if page else source
        content = document.page_content.strip()
        block = f"Source: {label}\n{content}"
        if total_chars + len(block) > settings.MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total_chars += len(block)
    return "\n\n".join(parts)


def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []
    for document in documents:
        key = document.metadata.get("chunk_id") or document.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        unique.append(document)
    return unique


def source_citations(documents: list[Document]) -> list[dict]:
    citations: list[dict] = []
    for document in dedupe_documents(documents):
        snippet = " ".join(document.page_content.split())[:240]
        citations.append(
            {
                "source": document.metadata.get("source", "unknown"),
                "page": document.metadata.get("page"),
                "chunk_id": document.metadata.get("chunk_id"),
                "doc_id": document.metadata.get("doc_id"),
                "snippet": snippet,
            }
        )
    return citations


def format_sources(documents: list[Document]) -> str:
    citations = source_citations(documents)
    if not citations:
        return ""
    lines = ["", "Sources:"]
    for citation in citations:
        page = f", page {citation['page']}" if citation.get("page") else ""
        lines.append(f"- {citation['source']}{page}: {citation['snippet']}")
    return "\n".join(lines)
