"""Tests for the local GPT-4o (tiktoken) adapter."""

from adapters import openai_tok


def test_returns_expected_shape():
    result = openai_tok.tokenize("Hello, world!")
    assert result["model"] == "gpt-4o"
    assert isinstance(result["tokens"], int)
    assert result["tokens"] > 0
    assert isinstance(result["pieces"], list)
    assert isinstance(result["token_ids"], list)
    assert result["tokens"] == len(result["pieces"]) == len(result["token_ids"])


def test_pieces_rejoin_to_original_text():
    # ASCII text is safe: every token boundary falls between whole characters,
    # so re-joining the decoded pieces reconstructs the input exactly.
    text = "The quick brown fox jumps over the lazy dog, 12345."
    result = openai_tok.tokenize(text)
    assert "".join(result["pieces"]) == text


def test_tokenizes_cjk_and_emoji_without_error():
    # Multi-byte UTF-8 input must be handled without exceptions.
    result = openai_tok.tokenize("自然语言处理 🚀💻 こんにちは")
    assert result["tokens"] > 0
    assert len(result["pieces"]) == result["tokens"]
