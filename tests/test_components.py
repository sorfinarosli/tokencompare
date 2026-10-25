"""Tests for the tie-aware winner / insight logic in components.py."""

import pytest

from components import chart_df, comparison_insights, compute_winners


def _row(model, tokens, chars_per_token, error=None):
    return {
        "model": model,
        "display_name": model,
        "tokens": tokens,
        "chars_per_token": chars_per_token,
        "cached": False,
        "error": error,
        "pieces": None,
        "token_ids": None,
    }


def test_compute_winners_handles_tie():
    rows = [_row("a", 14, 10.0), _row("b", 15, 9.0), _row("c", 14, 10.0)]
    winners = compute_winners(rows)
    # Two models tie for fewest tokens -> both win, neither is dropped.
    assert winners["fewest_tokens"] == ["a", "c"]


def test_compute_winners_single_winner():
    rows = [_row("a", 14, 10.0), _row("b", 15, 9.0), _row("c", 20, 6.0)]
    winners = compute_winners(rows)
    assert winners["fewest_tokens"] == ["a"]
    assert winners["most_efficient"] == ["a"]


def test_compute_winners_excludes_errors():
    rows = [_row("a", 14, 10.0), _row("b", 5, 20.0, error="boom"), _row("c", 20, 6.0)]
    winners = compute_winners(rows)
    assert "b" not in winners["fewest_tokens"]


def test_insights_mentions_tie_and_both_names():
    rows = [_row("a", 14, 10.0), _row("b", 14, 10.0), _row("c", 15, 9.0)]
    text = " ".join(comparison_insights(rows))
    assert "fewest tokens" in text
    assert "a" in text and "b" in text  # both tied models are named


def test_insights_empty_when_no_valid_rows():
    rows = [_row("a", None, None, error="boom")]
    assert comparison_insights(rows) == []


def test_chart_df_relative_percentage():
    rows = [_row("a", 14, 10.0), _row("b", 21, 6.0), _row("c", 14, 10.0)]
    df = chart_df(rows, multiplier=1)
    rel = dict(zip(df["Model"], df["Relative (%)"]))
    # Minimum is 14, so a and c are 0% and b is +50%.
    assert rel["a"] == 0.0
    assert rel["c"] == 0.0
    assert rel["b"] == pytest.approx(50.0)


def test_chart_df_excludes_missing_tokens():
    rows = [_row("a", 14, 10.0), _row("b", None, None, error="boom")]
    df = chart_df(rows, multiplier=1)
    assert list(df["Model"]) == ["a"]  # the erroring model is dropped
