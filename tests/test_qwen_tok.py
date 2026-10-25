"""Tests for the local Qwen 2.5 adapter.

The tokenizer is mocked so the tests run offline (no Hugging Face download).
"""

from adapters import qwen_tok


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [100, 200, 300]

    def decode(self, ids):
        return {100: "Hel", 200: "lo ", 300: "world"}[ids[0]]


def test_qwen_shape(monkeypatch):
    monkeypatch.setattr(qwen_tok, "_get_tokenizer", lambda: _FakeTokenizer())
    result = qwen_tok.tokenize("Hello world")
    assert result["model"] == "qwen2.5-0.5b"
    assert result["tokens"] == 3
    assert result["pieces"] == ["Hel", "lo ", "world"]
    assert result["token_ids"] == [100, 200, 300]
