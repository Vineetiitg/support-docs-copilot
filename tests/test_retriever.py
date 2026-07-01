import asyncio
from unittest.mock import AsyncMock, patch

from app.engine.query_transform import normalize_query, query_variants


def test_query_normalization_collapses_whitespace():
    assert normalize_query("  reset   password \n now ") == "reset password now"


@patch("app.engine.query_transform.ChatOpenAI")
def test_query_variants_add_helpful_expansions(mock_chat):
    class MockResult:
        content = '{"variants": ["troubleshoot error 404 steps"]}'
    
    from langchain_core.runnables import RunnableLambda
    mock_chat.return_value = RunnableLambda(lambda *args, **kwargs: MockResult())

    variants = asyncio.run(query_variants("How to fix error 404?"))

    assert "How to fix error 404?" in variants
    # The LLM mock adds "troubleshoot error 404 steps"
    assert "troubleshoot error 404 steps" in variants
