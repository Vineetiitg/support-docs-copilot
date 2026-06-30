from unittest.mock import patch

from app.engine.query_transform import normalize_query, query_variants


def test_query_normalization_collapses_whitespace():
    assert normalize_query("  reset   password \n now ") == "reset password now"


@patch("app.engine.query_transform.ChatOllama")
def test_query_variants_add_helpful_expansions(mock_chat):
    mock_instance = mock_chat.return_value
    class MockResult:
        content = '{"variants": ["troubleshoot error 404 steps"]}'
    
    mock_chain_invoke = mock_instance.invoke
    mock_chain_invoke.return_value = MockResult()
    
    # We also have to mock the prompt | llm chain, which returns a RunnableSequence
    # A simpler way is to mock the chain.invoke, but it's built inline. 
    # Let's mock ChatOllama.invoke to return the expected json if it's called directly by prompt | llm? No, ChatOllama gets passed prompt string.
    # We can patch ChatOllama.invoke
    mock_instance.invoke.return_value = MockResult()

    variants = query_variants("How to fix error 404?")

    assert "How to fix error 404?" in variants
    # The LLM mock adds "troubleshoot error 404 steps"
    assert "troubleshoot error 404 steps" in variants
