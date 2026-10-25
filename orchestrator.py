"""Request orchestrator with a SQLite-backed cache.

This module implements the PROCESSING stage of the approved architecture::

    INPUT       -> user text
    PROCESSING  -> check cache -> hit: return stored result (no API cost)
                               -> miss: call all tokenizers, store result
    OUTPUT      -> results handed back to the Streamlit dashboard

Each model's adapter is called inside a try/except so a failing provider (bad
key, rate limit, ...) produces an error entry for that model only and never
crashes the others.
"""

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

from adapters import gemini_tok, openai_tok, qwen_tok


def _default_db_path() -> Path:
    """Location of the SQLite cache.

    The cache is stored in a writable temp directory so it also works on
    read-only / ephemeral deployments (e.g. Streamlit Community Cloud). Set the
    ``TOKENCOMPARE_CACHE_DB`` environment variable to override the location.
    """
    override = os.environ.get("TOKENCOMPARE_CACHE_DB")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "tokencompare_cache.db"


#: Path to the SQLite cache database.
DB_PATH = _default_db_path()

#: The tokenizer adapters, in the order they are compared. Each module exposes
#: ``MODEL`` (str) and ``tokenize(text) -> dict``.
#:
#: Claude was replaced by Qwen 2.5: Claude's API has no free tier, whereas the
#: Qwen 2.5 tokenizer runs locally at zero cost (see adapters/claude_tok.py
#: for the original, disabled adapter).
ADAPTERS = [openai_tok, gemini_tok, qwen_tok]

#: Canonical model names, derived from the adapters (single source of truth).
MODEL_NAMES = [adapter.MODEL for adapter in ADAPTERS]


class RequestOrchestrator:
    """Runs tokenization across all models, caching results in SQLite.

    Cached results are keyed by ``sha256(model | text)`` so the exact same
    (model, text) pair is only ever sent to a provider once. The instance keeps
    running counters of cache hits and misses so the dashboard can report the
    cache hit rate.
    """

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.cache_hits = 0
        self.cache_misses = 0
        self._init_db()

    # ---------------------------------------------------------------- cache --

    @contextmanager
    def _connect(self):
        """Open a cache connection; commit on success and always close it."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create the cache table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key        TEXT PRIMARY KEY,
                    model      TEXT NOT NULL,
                    result     TEXT NOT NULL,   -- JSON-encoded adapter result
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @staticmethod
    def _cache_key(model: str, text: str) -> str:
        """Deterministic cache key for a (model, text) pair."""
        raw = f"{model}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        """Return the cached result for ``key``, or ``None`` on a miss."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_put(self, key: str, model: str, result: dict) -> None:
        """Store ``result`` (the adapter's raw dict) under ``key``."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, model, result) VALUES (?, ?, ?)",
                (key, model, json.dumps(result)),
            )

    def clear_cache(self) -> None:
        """Delete every cached entry and reset the hit/miss counters."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cache")
        self.cache_hits = 0
        self.cache_misses = 0

    # ----------------------------------------------------------- orchestrate --

    def compare(self, text: str) -> list[dict]:
        """Tokenize ``text`` with every adapter, using the cache where possible.

        Returns one dict per model. Each dict has the adapter's fields plus two
        extra fields added here::

            "cached" -- True if the result came from the cache, else False
            "error"  -- None on success, otherwise a human-readable message
        """
        results = []
        for adapter in ADAPTERS:
            model = adapter.MODEL
            key = self._cache_key(model, text)

            cached = self._cache_get(key)
            if cached is not None:
                # Cache hit: return the stored result at zero API cost.
                self.cache_hits += 1
                results.append({**cached, "cached": True, "error": None})
                continue

            # Cache miss: we must call the provider.
            self.cache_misses += 1
            try:
                result = adapter.tokenize(text)
            except Exception as exc:  # noqa: BLE001 -- one failing provider must not break the others
                results.append(
                    {
                        "model": model,
                        "tokens": None,
                        "pieces": None,
                        "token_ids": None,
                        "cached": False,
                        "error": str(exc),
                    }
                )
                continue

            # Only successful results are cached; failures are retried next time.
            self._cache_put(key, model, result)
            results.append({**result, "cached": False, "error": None})

        return results

    # -------------------------------------------------------------- metrics --

    @property
    def hit_rate(self) -> float:
        """Fraction of cache lookups that were hits (0.0 when no lookups yet)."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0
