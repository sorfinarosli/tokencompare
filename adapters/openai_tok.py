"""GPT-4o tokenization adapter (tiktoken, fully local, no API key).

GPT-4o's tokenizer (``o200k_base``) ships inside the ``tiktoken`` package, so we
can tokenize entirely offline. Because we do the work ourselves we also get the
individual token pieces (and their vocabulary IDs), which lets the dashboard
render them as chips with tooltips.
"""

import tiktoken

#: Canonical model identifier. Used in the cache key and must match the key in
#: ``pricing.yaml``.
MODEL = "gpt-4o"

#: The tiktoken encoding name used by GPT-4o.
ENCODING_NAME = "o200k_base"

#: Module-level cache so the encoding is only built once per process.
_encoding = None


def _get_encoding():
    """Return the shared tiktoken encoding, creating it on first use."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(ENCODING_NAME)
    return _encoding


def tokenize(text: str) -> dict:
    """Tokenize ``text`` with the GPT-4o encoding.

    Returns the token count, the decoded pieces, and the raw vocabulary IDs for
    each token. Each piece is produced by decoding the raw bytes of a single
    token back to text; ``errors="replace"`` guards against the (uncommon but
    possible) case where a token ends midway through a multi-byte UTF-8
    sequence.
    """
    enc = _get_encoding()
    token_ids = enc.encode(text)
    pieces = [
        enc.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
        for token_id in token_ids
    ]
    return {
        "model": MODEL,
        "tokens": len(token_ids),
        "pieces": pieces,
        "token_ids": token_ids,
    }
