"""Tests for the cost estimation module."""

import pytest

from cost import (
    display_name,
    estimate,
    get_model_info,
    has_price,
    is_open_source,
    load_prices,
    price_per_million,
)
from orchestrator import MODEL_NAMES


def test_estimate_hand_worked_values():
    prices = {
        "gpt-4o": {"input_per_million": 2.5},
        "gemini-3.8-flash": {"input_per_million": 0.1},
        "qwen2.5-0.5b": {"input_per_million": 0.0},
    }
    assert estimate("gpt-4o", 1_000_000, prices=prices) == pytest.approx(2.5)
    assert estimate("gpt-4o", 500_000, prices=prices) == pytest.approx(1.25)
    assert estimate("gemini-3.8-flash", 1_000, prices=prices) == pytest.approx(0.0001)
    assert estimate("gpt-4o", 0, prices=prices) == 0.0


def test_estimate_unknown_model_is_zero():
    assert estimate("does-not-exist", 1_000, prices={}) == 0.0


def test_pricing_keys_match_adapter_models():
    assert set(load_prices()) == set(MODEL_NAMES)


def test_pricing_structure_complete():
    # Every model must carry the fields the UI reads.
    for model in MODEL_NAMES:
        info = get_model_info(model)
        for field in ("display_name", "type", "input_per_million", "price_source", "price_date"):
            assert field in info, f"{model} is missing '{field}'"


def test_display_name():
    assert display_name("qwen2.5-0.5b") == "Qwen 2.5"


def test_open_source_detection():
    assert is_open_source("qwen2.5-0.5b") is True
    assert is_open_source("gpt-4o") is False
    assert is_open_source("gemini-3.8-flash") is False


def test_has_price_logic():
    prices = {
        "gpt-4o": {"type": "api", "input_per_million": 2.5},
        "gemini-3.8-flash": {"type": "api", "input_per_million": 0.0},
        "qwen2.5-0.5b": {"type": "open-source", "input_per_million": 0.0},
    }
    assert has_price("gpt-4o", prices=prices) is True
    assert has_price("gemini-3.8-flash", prices=prices) is False  # 0 -> not set
    assert has_price("qwen2.5-0.5b", prices=prices) is True  # free


def test_price_per_million():
    assert price_per_million("qwen2.5-0.5b") == 0.0
