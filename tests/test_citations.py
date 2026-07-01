from langchain_core.documents import Document

from app.engine.context_builder import build_context, source_citations


def test_source_citations_include_metadata_and_snippet():
    docs = [
        Document(
            page_content="Reset the router and verify DNS configuration.",
            metadata={"source": "runbook.md", "page": 2, "chunk_id": "abc", "doc_id": "doc-1"},
        )
    ]

    citations = source_citations(docs)

    assert citations[0]["source"] == "runbook.md"
    assert citations[0]["page"] == 2
    assert citations[0]["chunk_id"] == "abc"
    assert "Reset the router" in citations[0]["snippet"]


def test_context_builder_labels_sources():
    docs = [Document(page_content="Known issue details.", metadata={"source": "faq.txt"})]

    assert "Source: faq.txt" in build_context(docs)
