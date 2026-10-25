"""Tests for the request orchestrator and its SQLite cache.

Gemini and Qwen are mocked (Gemini needs an API key, Qwen needs a model
download); GPT-4o runs for real via tiktoken, which is entirely local.
"""

import pytest

from orchestrator import RequestOrchestrator
from adapters import gemini_tok, qwen_tok


@pytest.fixture
def orchestrator(tmp_path):
    """A fresh orchestrator backed by a temporary SQLite file."""
    return RequestOrchestrator(db_path=tmp_path / "test_cache.db")


@pytest.fixture
def fake_remote_adapters(monkeypatch):
    """Replace Gemini and Qwen with deterministic local fakes."""
    monkeypatch.setattr(
        gemini_tok,
        "tokenize",
        lambda text: {"model": gemini_tok.MODEL, "tokens": 7, "pieces": None},
    )
    monkeypatch.setattr(
        qwen_tok,
        "tokenize",
        lambda text: {"model": qwen_tok.MODEL, "tokens": 9, "pieces": None},
    )


def test_cache_hit_on_second_call(orchestrator, fake_remote_adapters):
    first = orchestrator.compare("hello world")
    assert all(r["cached"] is False for r in first)

    second = orchestrator.compare("hello world")
    assert all(r["cached"] is True for r in second)
    # Each of the three models is now served from cache.
    assert orchestrator.cache_hits == 3
    assert orchestrator.cache_misses == 3


def test_one_failure_does_not_break_others(orchestrator, monkeypatch):
    def boom(text):
        raise RuntimeError("invalid API key")

    monkeypatch.setattr(gemini_tok, "tokenize", boom)
    monkeypatch.setattr(
        qwen_tok,
        "tokenize",
        lambda text: {"model": qwen_tok.MODEL, "tokens": 4, "pieces": None},
    )

    results = orchestrator.compare("some text")
    by_model = {r["model"]: r for r in results}

    # GPT-4o (local) and the mocked Qwen succeed.
    assert by_model["gpt-4o"]["tokens"] is not None
    assert by_model["gpt-4o"]["error"] is None
    assert by_model["qwen2.5-0.5b"]["tokens"] == 4
    assert by_model["qwen2.5-0.5b"]["error"] is None
    # Gemini fails, but the failure is isolated to its own entry.
    assert by_model["gemini-3.8-flash"]["tokens"] is None
    assert by_model["gemini-3.8-flash"]["error"] == "invalid API key"


def test_clear_cache_resets_counters(orchestrator, fake_remote_adapters):
    orchestrator.compare("hello")
    orchestrator.clear_cache()
    assert orchestrator.cache_hits == 0
    assert orchestrator.cache_misses == 0
    # After clearing, the same text is a cache miss again.
    results = orchestrator.compare("hello")
    assert all(r["cached"] is False for r in results)


def test_hit_rate(orchestrator, fake_remote_adapters):
    assert orchestrator.hit_rate == 0.0  # no lookups yet
    orchestrator.compare("hello")
    orchestrator.compare("hello")
    # 3 misses then 3 hits -> 50%.
    assert orchestrator.hit_rate == pytest.approx(0.5)
