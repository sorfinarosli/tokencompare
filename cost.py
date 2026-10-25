"""Cost estimation.

Loads per-model pricing from ``pricing.yaml`` and converts token counts into
estimated USD costs. Supports both commercial (API) models and free,
open-source models.

``pricing.yaml`` gives each model::

    display_name      -> human-friendly name shown in the UI
    type              -> "api" or "open-source"
    input_per_million -> USD price per 1,000,000 input tokens
    price_source      -> official pricing page URL
    price_date        -> date the price was last checked
"""

from pathlib import Path

import yaml

#: Location of the pricing file, resolved relative to this module.
PRICING_PATH = Path(__file__).resolve().parent / "pricing.yaml"

#: The two model types recognised by the system.
TYPE_API = "api"
TYPE_OPEN_SOURCE = "open-source"


def load_prices() -> dict:
    """Load ``pricing.yaml`` and return ``{model: {...}}``."""
    with open(PRICING_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_model_info(model: str, prices: dict | None = None) -> dict:
    """Return the full pricing entry for ``model`` (``{}`` if unknown)."""
    if prices is None:
        prices = load_prices()
    return prices.get(model) or {}


def model_type(model: str, prices: dict | None = None) -> str:
    """Return the model's ``type`` ("api", "open-source", or "" if unknown)."""
    return get_model_info(model, prices).get("type", "")


def display_name(model: str, prices: dict | None = None) -> str:
    """Return the human-friendly display name for a model."""
    return get_model_info(model, prices).get("display_name", model)


def is_open_source(model: str, prices: dict | None = None) -> bool:
    """True when ``model`` is a free, open-source model."""
    return model_type(model, prices) == TYPE_OPEN_SOURCE


def price_per_million(model: str, prices: dict | None = None) -> float:
    """USD price per 1M input tokens (0.0 for open-source or unset)."""
    return float(get_model_info(model, prices).get("input_per_million", 0.0))


def has_price(model: str, prices: dict | None = None) -> bool:
    """True when the model has a usable price.

    Open-source models are free, so they always count as priced. A commercial
    model only counts as priced when its per-million price is greater than 0.
    """
    if is_open_source(model, prices):
        return True
    return price_per_million(model, prices) > 0.0


def estimate(model: str, tokens: int, prices: dict | None = None) -> float:
    """Estimated USD cost of ``tokens`` input tokens for ``model``.

    ``cost = tokens / 1_000_000 * price_per_million``.
    """
    return tokens / 1_000_000 * price_per_million(model, prices)
