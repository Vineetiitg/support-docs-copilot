"""
Unit tests for the LangGraph Self-RAG workflow.

These tests mock all LLM calls and external retrieval so they run
instantly without network access, API keys, or a running Qdrant instance.

Strategy: patch the individual async node functions (retrieve, grade_documents,
generate, evaluate_answer) rather than the raw LLM objects, because LangGraph
compiles the graph at import time and the pipe operator is hard to intercept.
"""

import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers – sample data
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    Document(
        page_content="To reset your password, go to Settings > Security > Reset Password.",
        metadata={"source": "faq.md", "doc_id": "faq-001", "chunk_id": "c1"},
    ),
    Document(
        page_content="Our support team is available 24/7 via the Help Center.",
        metadata={"source": "contact.md", "doc_id": "contact-001", "chunk_id": "c2"},
    ),
]

SAMPLE_SOURCES = [
    {"source": "faq.md", "page": None, "chunk_id": "c1", "doc_id": "faq-001",
     "snippet": "To reset your password, go to Settings > Security > Reset Password."},
]


def _base_state(**overrides):
    """Build a minimal valid input state for the workflow."""
    state = {"question": "How do I reset my password?", "chat_history": [], "run_count": 0}
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Fake node return values
# ---------------------------------------------------------------------------

async def _fake_retrieve(state):
    return {
        "documents": SAMPLE_DOCS,
        "sources": SAMPLE_SOURCES,
        "question": state["question"],
        "run_count": state.get("run_count", 0),
    }


async def _fake_retrieve_empty(state):
    return {
        "documents": [],
        "sources": [],
        "question": state["question"],
        "run_count": state.get("run_count", 0),
    }


async def _fake_grade_all_relevant(state):
    return {"documents": state.get("documents", [])}


async def _fake_grade_all_irrelevant(state):
    return {"documents": []}


async def _fake_generate(state):
    run_count = state.get("run_count", 0) + 1
    return {
        "generation": "Reset your password in Settings > Security.",
        "sources": SAMPLE_SOURCES,
        "run_count": run_count,
    }


async def _fake_evaluate_grounded(state):
    return {"grounded": "yes", "confidence_score": 0.95}


async def _fake_evaluate_hallucinated(state):
    return {"grounded": "no", "confidence_score": 0.2}


# ---------------------------------------------------------------------------
# Tests – Happy Path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_happy_path_returns_grounded_answer():
    """Full graph: retrieve → grade(yes) → generate → evaluate(grounded) → END."""
    with patch("app.graph.workflow.retrieve", new=_fake_retrieve), \
         patch("app.graph.workflow.grade_documents", new=_fake_grade_all_relevant), \
         patch("app.graph.workflow.generate", new=_fake_generate), \
         patch("app.graph.workflow.evaluate_answer", new=_fake_evaluate_grounded):

        from app.graph.workflow import compile_workflow
        agent = compile_workflow()

        state = await agent.ainvoke(_base_state())

    assert state["generation"] == "Reset your password in Settings > Security."
    assert state["confidence_score"] == 0.95
    assert state["grounded"] == "yes"
    assert state["run_count"] == 1
    assert len(state["sources"]) > 0


# ---------------------------------------------------------------------------
# Tests – No Documents Retrieved
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_ends_when_no_documents_retrieved():
    """When retriever returns nothing, grade filters to empty → graph ends."""
    with patch("app.graph.workflow.retrieve", new=_fake_retrieve_empty), \
         patch("app.graph.workflow.grade_documents", new=_fake_grade_all_irrelevant):

        from app.graph.workflow import compile_workflow
        agent = compile_workflow()

        state = await agent.ainvoke(_base_state(question="What is the meaning of life?"))

    # No docs → no generation
    assert "generation" not in state or state.get("generation") is None
    assert state["documents"] == []


# ---------------------------------------------------------------------------
# Tests – All Documents Graded Irrelevant
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_ends_when_all_docs_graded_irrelevant():
    """When grader rejects all docs, graph ends without calling generate."""
    with patch("app.graph.workflow.retrieve", new=_fake_retrieve), \
         patch("app.graph.workflow.grade_documents", new=_fake_grade_all_irrelevant):

        from app.graph.workflow import compile_workflow
        agent = compile_workflow()

        state = await agent.ainvoke(_base_state(question="Tell me about quantum physics"))

    assert "generation" not in state or state.get("generation") is None


# ---------------------------------------------------------------------------
# Tests – Hallucination Retry Loop Caps at 3
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_retries_on_hallucination_and_caps_at_max():
    """
    Evaluator always says 'hallucinated', so the graph loops back to
    generate. After run_count reaches 3, check_hallucinations routes to END.
    """
    generate_call_count = 0

    async def _counting_generate(state):
        nonlocal generate_call_count
        generate_call_count += 1
        run_count = state.get("run_count", 0) + 1
        return {
            "generation": f"Attempt {generate_call_count}",
            "sources": SAMPLE_SOURCES,
            "run_count": run_count,
        }

    with patch("app.graph.workflow.retrieve", new=_fake_retrieve), \
         patch("app.graph.workflow.grade_documents", new=_fake_grade_all_relevant), \
         patch("app.graph.workflow.generate", new=_counting_generate), \
         patch("app.graph.workflow.evaluate_answer", new=_fake_evaluate_hallucinated):

        from app.graph.workflow import compile_workflow
        agent = compile_workflow()

        state = await agent.ainvoke(_base_state())

    # Must stop at or before 3 retries
    assert state["run_count"] <= 3, f"Expected max 3 retries, got {state['run_count']}"
    assert generate_call_count <= 3, f"Generate called {generate_call_count} times, expected ≤ 3"
    assert state["generation"], "Should return the last attempt's answer"


# ---------------------------------------------------------------------------
# Tests – Single Retry Then Grounded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_retries_once_then_succeeds():
    """First evaluation says hallucinated, second says grounded → stops at run_count=2."""
    eval_call_count = 0

    async def _eval_fail_then_pass(state):
        nonlocal eval_call_count
        eval_call_count += 1
        if eval_call_count == 1:
            return {"grounded": "no", "confidence_score": 0.3}
        return {"grounded": "yes", "confidence_score": 0.9}

    with patch("app.graph.workflow.retrieve", new=_fake_retrieve), \
         patch("app.graph.workflow.grade_documents", new=_fake_grade_all_relevant), \
         patch("app.graph.workflow.generate", new=_fake_generate), \
         patch("app.graph.workflow.evaluate_answer", new=_eval_fail_then_pass):

        from app.graph.workflow import compile_workflow
        agent = compile_workflow()

        state = await agent.ainvoke(_base_state())

    assert state["grounded"] == "yes"
    assert state["confidence_score"] == 0.9
    assert eval_call_count == 2


# ---------------------------------------------------------------------------
# Tests – Output State Structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_output_state_has_required_keys():
    """Verify the final state dict contains all expected keys after a full run."""
    with patch("app.graph.workflow.retrieve", new=_fake_retrieve), \
         patch("app.graph.workflow.grade_documents", new=_fake_grade_all_relevant), \
         patch("app.graph.workflow.generate", new=_fake_generate), \
         patch("app.graph.workflow.evaluate_answer", new=_fake_evaluate_grounded):

        from app.graph.workflow import compile_workflow
        agent = compile_workflow()

        state = await agent.ainvoke(_base_state())

    required_keys = {"question", "documents", "generation", "sources", "run_count",
                     "confidence_score", "grounded"}
    assert required_keys.issubset(state.keys()), f"Missing keys: {required_keys - state.keys()}"


# ---------------------------------------------------------------------------
# Tests – Chat History Preservation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_preserves_chat_history():
    """Ensure chat_history is passed through the graph without corruption."""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]

    with patch("app.graph.workflow.retrieve", new=_fake_retrieve), \
         patch("app.graph.workflow.grade_documents", new=_fake_grade_all_relevant), \
         patch("app.graph.workflow.generate", new=_fake_generate), \
         patch("app.graph.workflow.evaluate_answer", new=_fake_evaluate_grounded):

        from app.graph.workflow import compile_workflow
        agent = compile_workflow()

        state = await agent.ainvoke(_base_state(chat_history=history))

    assert state["chat_history"] == history, "Chat history should be preserved"


# ---------------------------------------------------------------------------
# Tests – Question Passthrough
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_returns_original_question():
    """The original question must be present in the final state unchanged."""
    question = "How do I contact support?"

    with patch("app.graph.workflow.retrieve", new=_fake_retrieve), \
         patch("app.graph.workflow.grade_documents", new=_fake_grade_all_relevant), \
         patch("app.graph.workflow.generate", new=_fake_generate), \
         patch("app.graph.workflow.evaluate_answer", new=_fake_evaluate_grounded):

        from app.graph.workflow import compile_workflow
        agent = compile_workflow()

        state = await agent.ainvoke(_base_state(question=question))

    assert state["question"] == question
