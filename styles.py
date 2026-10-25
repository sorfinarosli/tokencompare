"""Centralised visual styling for the dashboard.

All CSS lives here (injected once via :func:`inject_css`) and the per-model
brand colours are defined here too, so the same colour is reused across the
metric cards, token chips, tables and Altair charts. Keeping styling in one
place keeps ``app.py`` readable.
"""

import streamlit as st

# --------------------------------------------------------------------------- #
# Model brand colours (one per model, used everywhere)
# --------------------------------------------------------------------------- #

#: Canonical model key -> brand colour (hex).
MODEL_COLORS = {
    "gpt-4o": "#10b981",          # green  (emerald)
    "gemini-3.8-flash": "#3b82f6",  # blue
    "qwen2.5-0.5b": "#8b5cf6",      # purple (violet)
}

#: Fallback for any model key we don't recognise.
DEFAULT_COLOR = "#64748b"  # slate


def model_color(model: str) -> str:
    """Return the brand colour for a model (slate fallback)."""
    return MODEL_COLORS.get(model, DEFAULT_COLOR)


# --------------------------------------------------------------------------- #
# Small colour helpers
# --------------------------------------------------------------------------- #

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def tint(hex_color: str, amount: float = 0.85) -> str:
    """Blend ``hex_color`` toward white by ``amount`` (0..1 -> lighter)."""
    r, g, b = _hex_to_rgb(hex_color)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return _rgb_to_hex((r, g, b))


def chip_colours(model: str) -> tuple[str, str]:
    """Two alternating light shades of a model's colour, for token chips."""
    base = model_color(model)
    return tint(base, 0.82), tint(base, 0.93)


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #

_CSS = """
<style>
/* --- base ---------------------------------------------------------------- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

/* --- hero header ---------------------------------------------------------- */
.hero { margin-bottom: 1.4rem; }
.hero-title { font-size: 2.1rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; margin: 0; }
.hero-subtitle { color: #475569; font-size: 1.02rem; margin-top: 0.3rem; }
.hero-badges { margin-top: 0.9rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
.badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 999px;
    padding: 0.28rem 0.8rem; font-size: 0.82rem; font-weight: 600; color: #334155;
}
.badge .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

/* --- bordered input container -------------------------------------------- */
.input-box {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 1.1rem 1.3rem 1.3rem; box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.char-counter { text-align: right; color: #94a3b8; font-size: 0.8rem; margin-top: 0.25rem; }

/* --- metric cards --------------------------------------------------------- */
.card-row {
    display: flex; flex-wrap: wrap; gap: 1rem; align-items: stretch; margin: 0.9rem 0 1.2rem;
}
.model-card {
    flex: 1 1 220px; min-width: 210px; max-width: 100%;
    display: flex; flex-direction: column;
    background: #ffffff; border-radius: 14px; padding: 1.1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.08), 0 6px 18px rgba(15,23,42,0.04);
    position: relative;
    transition: transform .15s ease, box-shadow .15s ease;
}
.model-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 26px rgba(15,23,42,0.12);
}
.card-head { margin-bottom: 0.6rem; }
.card-title { font-size: 1rem; font-weight: 700; line-height: 1.2; }
.card-version { font-size: 0.75rem; color: #94a3b8; font-family: 'Consolas', 'Courier New', monospace; margin-top: 0.15rem; }
.card-tokens { font-size: 2.4rem; font-weight: 800; color: #0f172a; line-height: 1.05; }
.card-unit { color: #64748b; font-size: 0.8rem; font-weight: 600; }
.card-meta { margin-top: 0.8rem; display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.85rem; color: #334155; }
.meta-row { display: flex; justify-content: space-between; gap: 0.5rem; }
.meta-label { color: #94a3b8; font-weight: 500; white-space: nowrap; }
.meta-value { font-weight: 700; text-align: right; }
.cost-source { font-size: 0.72rem; color: #94a3b8; margin-top: 0.35rem; }
.card-error { font-size: 0.85rem; color: #b91c1c; font-weight: 600; margin: 0.3rem 0; }
.card-foot { margin-top: auto; padding-top: 0.7rem; }
.pill {
    display: inline-block; margin-top: 0.7rem; border-radius: 999px;
    padding: 0.15rem 0.6rem; font-size: 0.72rem; font-weight: 700;
}
.pill-cached { background: #dcfce7; color: #166534; }
.pill-live { background: #eef2ff; color: #3730a3; }
.winner-badge {
    display: inline-block; margin-top: 0.35rem; margin-right: 0.3rem;
    border-radius: 999px; padding: 0.12rem 0.55rem; font-size: 0.7rem; font-weight: 700;
    background: #fef3c7; color: #92400e; border: 1px solid #fde68a;
}

/* --- error card ----------------------------------------------------------- */
.error-card {
    background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px;
    padding: 0.9rem 1.1rem; color: #7f1d1d; font-size: 0.9rem;
}
.error-card b { display: block; margin-bottom: 0.2rem; }

/* --- token chips ---------------------------------------------------------- */
.chip-wrap { display: flex; flex-wrap: wrap; gap: 4px; line-height: 2; padding: 0.5rem 0; }
.token-chip {
    display: inline-block; border-radius: 5px; padding: 0.12rem 0.5rem;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 0.85rem; border: 1px solid; cursor: default;
}
.no-pieces {
    background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px;
    padding: 0.7rem 0.8rem; font-size: 0.82rem; color: #64748b; margin-top: 0.3rem;
}

/* --- empty state ---------------------------------------------------------- */
.empty-state {
    background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 14px;
    padding: 2rem; text-align: center; color: #475569; margin: 1rem 0;
}
.step {
    display: inline-block; vertical-align: top; width: 210px; margin: 0.5rem 0.8rem;
    text-align: center;
}
.step-num {
    width: 34px; height: 34px; border-radius: 50%; background: #4f46e5; color: #fff;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 800; margin-bottom: 0.4rem;
}
.step-title { font-weight: 700; color: #0f172a; }
.step-desc { font-size: 0.85rem; color: #64748b; }

/* --- comparison insights --------------------------------------------------- */
.insights-box {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1rem 1.2rem; margin: 0.6rem 0 1rem;
}
.insights-title { font-weight: 700; color: #0f172a; margin-bottom: 0.5rem; font-size: 1rem; }
.insight { color: #334155; font-size: 0.92rem; margin: 0.35rem 0; line-height: 1.5; }
.insight-note { color: #94a3b8; font-size: 0.8rem; margin-top: 0.7rem; font-style: italic; }
.section-title { font-size: 1.1rem; font-weight: 700; color: #0f172a; margin: 1.1rem 0 0.6rem; }

/* --- sidebar tweaks ------------------------------------------------------- */
.sidebar-section { margin-bottom: 1rem; }
.sidebar-section .title { font-weight: 700; color: #0f172a; margin-bottom: 0.3rem; }
.pricing-line { font-size: 0.78rem; color: #475569; margin: 0.15rem 0; }
</style>
"""


def inject_css() -> None:
    """Inject the dashboard's CSS once."""
    st.markdown(_CSS, unsafe_allow_html=True)
