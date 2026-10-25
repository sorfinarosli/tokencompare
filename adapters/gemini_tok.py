"""Gemini tokenization adapter (Google GenAI SDK).

Uses the ``count_tokens`` endpoint of the ``google-genai`` SDK. Gemini only
reports a token count (it does not expose the individual token pieces), so the
adapter returns ``pieces=None``.
"""

import os

from google import genai

#: Canonical model identifier. Change this to point at a different Gemini model
#: if you prefer; keep the same key in ``pricing.yaml``.
MODEL = "gemini-3.8-flash"

#: Module-level cache so the client is only built once per process.
_client = None


def _api_key() -> str | None:
    """Read ``GEMINI_API_KEY`` from Streamlit secrets, then the environment.

    Streamlit secrets take precedence so the key can live in
    ``.streamlit/secrets.toml`` (which is git-ignored). The environment
    fallback lets the adapter work outside Streamlit too.
    """
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def _get_client():
    """Return a shared Google GenAI client, creating it on first use."""
    global _client
    if _client is None:
        key = _api_key()
        if not key or key.startswith("your-"):
            raise ValueError(
                "GEMINI_API_KEY is not configured. Add it to "
                ".streamlit/secrets.toml (see .streamlit/secrets.toml.example)."
            )
        _client = genai.Client(api_key=key)
    return _client


def tokenize(text: str) -> dict:
    """Return the number of tokens Gemini would use for ``text``.

    Gemini only reports a count, so ``pieces`` and ``token_ids`` are ``None``.
    """
    client = _get_client()
    response = client.models.count_tokens(model=MODEL, contents=text)
    return {
        "model": MODEL,
        "tokens": response.total_tokens,
        "pieces": None,
        "token_ids": None,
    }
