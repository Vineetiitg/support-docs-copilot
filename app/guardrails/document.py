from langchain_core.documents import Document

from app.guardrails.input import PROMPT_INJECTION_PATTERNS


def document_has_malicious_instruction(document: Document) -> bool:
    normalized = document.page_content.lower()
    return any(pattern in normalized for pattern in PROMPT_INJECTION_PATTERNS)


def filter_malicious_documents(documents: list[Document]) -> tuple[list[Document], list[str]]:
    safe: list[Document] = []
    flagged: list[str] = []
    for document in documents:
        if document_has_malicious_instruction(document):
            flagged.append(document.metadata.get("source", "unknown"))
            continue
        safe.append(document)
    return safe, flagged
