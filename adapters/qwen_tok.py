"""Qwen 2.5 tokenization adapter (local, open-source, no API key).

Qwen 2.5 is an open-weight model family from Alibaba. Its tokenizer is loaded
locally through Hugging Face ``transformers``, so this adapter needs no API
key, no account and no payment.

Because the tokenizer runs locally we can also return the individual token
pieces and their vocabulary IDs, so the dashboard renders Qwen 2.5's token
breakdown alongside GPT-4o's.
"""

from transformers import AutoTokenizer

#: Canonical model identifier. Used in the cache key and must match the key in
#: ``pricing.yaml``.
MODEL = "qwen2.5-0.5b"

#: Hugging Face repository. Qwen 2.5 uses the *same* tokenizer across every
#: model size, so we load the smallest (0.5B). Only the tokenizer files (a few
#: MB) are downloaded on first use — never the model weights.
HF_REPO = "Qwen/Qwen2.5-0.5B"

#: Module-level cache so the tokenizer is only downloaded/loaded once.
_tokenizer = None


def _get_tokenizer():
    """Return the shared Qwen 2.5 tokenizer, loading it on first use."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    return _tokenizer


def tokenize(text: str) -> dict:
    """Return the tokens, decoded pieces, and vocabulary IDs for ``text``.

    ``tokenizer.decode([id])`` decodes a single token ID back to its text; for
    Qwen's byte-level BPE this reconstructs the substring (including any
    leading space), so the pieces re-join to the original text.
    """
    tok = _get_tokenizer()
    token_ids = tok.encode(text, add_special_tokens=False)
    pieces = [tok.decode([token_id]) for token_id in token_ids]
    return {
        "model": MODEL,
        "tokens": len(token_ids),
        "pieces": pieces,
        "token_ids": token_ids,
    }
