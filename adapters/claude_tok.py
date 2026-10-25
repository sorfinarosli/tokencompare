"""Claude tokenization adapter (Anthropic SDK) — DISABLED.

This adapter is kept for reference only. Claude's API has no free tier, so the
free Qwen 2.5 tokenizer (``qwen_tok.py``) replaces it in the comparison. This
module is NOT imported by ``adapters/__init__.py`` and NOT listed in the
orchestrator's ``ADAPTERS``.

If re-enabled, it uses the ``count_tokens`` endpoint of the ``anthropic`` SDK.
Claude only reports a token count (it does not expose the individual token
pieces), so the adapter returns ``pieces=None``.
"""

import os

import anthropic

#: Canonical model identifier. Change this if you use a different Claude model;
#: keep the same key in ``pricing.yaml``.
MODEL = "claude-sonnet-5"

#: Module-level cache so the client is only built once per process.
_client = None

#: Claude's message-formatting overhead (see :func:`_message_overhead`). Computed
#: lazily once and cached, because it is constant for a fixed model.
_overhead = None


def _api_key() -> str | None:
    """Read ``ANTHROPIC_API_KEY`` from Streamlit secrets, then the environment.

    Streamlit secrets take precedence so the key can live in
    ``.streamlit/secrets.toml`` (which is git-ignored). The environment
    fallback lets the adapter work outside Streamlit too.
    """
    try:
        import streamlit as st

        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def _get_client():
    """Return a shared Anthropic client, creating it on first use."""
    global _client
    if _client is None:
        key = _api_key()
        if not key or key.startswith("your-"):
            raise ValueError(
                "ANTHROPIC_API_KEY is not configured. Add it to "
                ".streamlit/secrets.toml (see .streamlit/secrets.toml.example)."
            )
        _client = anthropic.Anthropic(api_key=key)
    return _client


def _message_overhead() -> int:
    """Return the token overhead Claude adds for message formatting.

    ``count_tokens`` counts not only the text itself but also a handful of
    structural tokens Claude inserts around each message (role markers, turn
    delimiters, etc.). That overhead is not part of the text the user typed,
    so to make the number comparable with tiktoken's raw count we subtract it.

    We measure the overhead once: a single ASCII character ("x") is exactly one
    token, so the count returned for a one-character message is
    ``1 + overhead``, and the overhead is therefore ``baseline - 1``.
    """
    global _overhead
    if _overhead is None:
        client = _get_client()
        baseline = client.messages.count_tokens(
            model=MODEL,
            messages=[{"role": "user", "content": "x"}],
        )
        _overhead = baseline.input_tokens - 1
    return _overhead


def tokenize(text: str) -> dict:
    """Return the number of tokens Claude would use for ``text`` (text only)."""
    client = _get_client()
    response = client.messages.count_tokens(
        model=MODEL,
        messages=[{"role": "user", "content": text}],
    )
    # Strip the message-formatting overhead so the count reflects the text only.
    tokens = max(0, response.input_tokens - _message_overhead())
    return {"model": MODEL, "tokens": tokens, "pieces": None}
