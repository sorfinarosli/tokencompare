"""Tokenizer adapters for the Multi-Model LLM Tokenization Comparison System.

Every adapter in this package exposes two things:

    MODEL      -- the canonical model identifier (str)
    tokenize   -- callable(text: str) -> dict

The dict returned by ``tokenize`` always has the same shape, regardless of the
provider::

    {
        "model":     str,                # the model identifier
        "tokens":    int,                # number of tokens for the input text
        "pieces":    list[str] | None,   # decoded token pieces (None when the
                                         # provider only reports a count)
        "token_ids": list[int] | None,   # vocabulary IDs per token (None when
                                         # the provider only reports a count)
    }

GPT-4o and Qwen 2.5 run locally, so they return ``pieces`` and ``token_ids``
(used for the token breakdown + chip tooltips). Gemini is count-only, so it
returns ``None`` for both.

Adapters raise exceptions on failure. It is the orchestrator's job to catch
those exceptions so a single broken provider never crashes the whole
comparison.
"""

from . import gemini_tok
from . import openai_tok
from . import qwen_tok

# NOTE: claude_tok.py is kept on disk for reference (the original paid Claude
# adapter), but it is intentionally NOT imported or used — Claude's API has no
# free tier, so the free Qwen 2.5 tokenizer replaces it in the comparison.

__all__ = ["openai_tok", "gemini_tok", "qwen_tok"]
