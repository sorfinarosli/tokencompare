"""Streamlit dashboard for the LLM Tokenization Comparison System.

This is the presentation layer. The heavy HTML/CSS/chart building lives in
``components.py`` and ``styles.py``; the tokenization logic lives in
``orchestrator.py`` and the ``adapters/`` package.
"""

import streamlit as st

from components import (
    build_rows,
    render_breakdown,
    render_cards,
    render_charts,
    render_cost_simulator,
    render_insights,
    render_table,
)
from cost import display_name, get_model_info, is_open_source, price_per_million
from orchestrator import MODEL_NAMES, RequestOrchestrator
from styles import inject_css, model_color

# --------------------------------------------------------------------------- #
# Page config, styles & orchestrator singleton
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="LLM Tokenization Comparison", layout="wide")
inject_css()


@st.cache_resource
def get_orchestrator() -> RequestOrchestrator:
    return RequestOrchestrator()


orchestrator = get_orchestrator()

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Cost display views: label -> multiplier applied to the single-request cost.
COST_BASIS = {
    "Single request": 1,
    "Per 1,000 requests": 1_000,
    "Per 1,000,000 requests": 1_000_000,
}

#: Quick-fill sample texts for the input box.
SAMPLE_TEXTS = {
    "English": "Tokenization is the process of splitting text into smaller units called tokens.",
    "Malay": "Pemprosesan bahasa semula jadi membolehkan komputer memahami bahasa manusia.",
    "Others": "自然语言处理是人工智能的一个重要领域。",
}

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def render_sidebar() -> int:
    """Render the sidebar; return the selected cost multiplier."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section"><div class="title">LLM Tokenization Comparison</div>'
            '<div style="font-size:0.8rem;color:#64748b;">'
            "A multi-model LLM tokenization comparison dashboard.</div></div>",
            unsafe_allow_html=True,
        )

        # --- cost basis ---
        st.markdown('<div class="sidebar-section"><div class="title">Cost basis</div>', unsafe_allow_html=True)
        cost_basis = st.selectbox(
            "Cost basis", list(COST_BASIS.keys()), index=0, key="cost_basis", label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # --- pricing info ---
        st.markdown('<div class="sidebar-section"><div class="title">Pricing</div>', unsafe_allow_html=True)
        for m in MODEL_NAMES:
            info = get_model_info(m)
            typ = info.get("type", "")
            src = info.get("price_source", "")
            date = info.get("price_date", "") or "not set"
            if typ == "open-source":
                label = "N/A (Local Tokenizer)"
            elif price_per_million(m) > 0:
                label = f"${price_per_million(m):,.2f} / 1M input tokens"
            else:
                label = "Price not set"
            st.markdown(
                f'<div class="pricing-line"><b>{display_name(m)}</b> — {label}<br>'
                f'<span style="color:#94a3b8;">{src or "no source"} · {date}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    return COST_BASIS[cost_basis]


# --------------------------------------------------------------------------- #
# Hero header
# --------------------------------------------------------------------------- #


def render_hero() -> None:
    badges = "".join(
        f'<span class="badge"><span class="dot" style="background:{model_color(m)};"></span>{display_name(m)}</span>'
        for m in MODEL_NAMES
    )
    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-title">LLM Tokenization Comparison</div>'
        f'<div class="hero-subtitle">Compare token counts, cost and efficiency of GPT-4o, '
        f'Gemini and Qwen 2.5 — side by side.</div>'
        f'<div class="hero-badges">{badges}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Empty state
# --------------------------------------------------------------------------- #


def render_empty_state() -> None:
    st.markdown(
        '<div class="empty-state">'
        '<div style="font-weight:700;font-size:1.1rem;color:#0f172a;">'
        "Compare how three LLMs tokenize the same text</div>"
        '<div class="step"><div class="step-num">1</div><div class="step-title">Type or paste text</div>'
        '<div class="step-desc">Any language.</div></div>'
        '<div class="step"><div class="step-num">2</div><div class="step-title">Click Compare</div>'
        '<div class="step-desc">GPT-4o, Gemini and Qwen 2.5 tokenize it.</div></div>'
        '<div class="step"><div class="step-num">3</div><div class="step-title">Explore results</div>'
        '<div class="step-desc">Tokens, cost, efficiency and charts.</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Input section
# --------------------------------------------------------------------------- #


def render_input_section() -> str:
    """Render the input box (text area, counter, samples, buttons)."""
    # Apply a pending input change (from a sample button or Clear) BEFORE the
    # text_area is instantiated. Streamlit forbids editing a widget's state
    # after the widget has already been rendered this run, so we defer the
    # change through a flag + rerun instead of touching "input_text" directly.
    pending = st.session_state.pop("_pending_input", None)
    if pending is not None:
        if pending == "":
            st.session_state.pop("last_results", None)
            st.session_state.pop("last_text", None)
        st.session_state["input_text"] = pending

    with st.container(border=True):
        text = st.text_area(
            "Text to tokenize",
            key="input_text",
            height=180,
            placeholder="Type or paste text, or click a sample below…",
        )
        st.markdown(f'<div class="char-counter">{len(text):,} characters</div>', unsafe_allow_html=True)
        st.caption("Try a sample:")
        sample_cols = st.columns(len(SAMPLE_TEXTS))
        for col, (label, sample) in zip(sample_cols, SAMPLE_TEXTS.items()):
            if col.button(label, key=f"sample_{label}", width="stretch"):
                st.session_state["_pending_input"] = sample
                st.rerun()
        b1, b2 = st.columns([1, 1])
        compare = b1.button("Compare", type="primary", width="stretch")
        clear = b2.button("Clear", width="stretch")

    if clear:
        st.session_state["_pending_input"] = ""
        st.rerun()
    if compare:
        if not text.strip():
            st.warning("Please enter some text before comparing.")
        else:
            with st.spinner("Comparing tokenizers…"):
                st.session_state["last_results"] = orchestrator.compare(text)
            st.session_state["last_text"] = text
    return text


# --------------------------------------------------------------------------- #
# Single-input tab
# --------------------------------------------------------------------------- #


def render_single_tab(cost_mult: int) -> None:
    render_empty_state()
    text = render_input_section()

    if "last_results" not in st.session_state:
        return

    if st.session_state.get("last_text") != text:
        st.info("The input has changed since the last comparison — click Compare to update.")

    rows = build_rows(st.session_state["last_results"], st.session_state["last_text"])

    # Warn about any API model that still has no price configured.
    unpriced = [
        r["display_name"]
        for r in rows
        if not r["error"] and not is_open_source(r["model"]) and not price_per_million(r["model"])
    ]
    if unpriced:
        st.warning(
            "No price configured for: " + ", ".join(unpriced)
            + " — showing 'Price not set'. Add it in pricing.yaml."
        )

    tab_overview, tab_breakdown, tab_charts, tab_table = st.tabs(
        ["Overview", "Token Breakdown", "Charts", "Data Table"]
    )
    with tab_overview:
        render_cards(rows, cost_mult)
        render_insights(rows)
    with tab_breakdown:
        render_breakdown(rows)
    with tab_charts:
        render_charts(rows, cost_mult)
        render_cost_simulator(rows)
    with tab_table:
        render_table(rows, cost_mult)


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #

cost_mult = render_sidebar()
render_hero()

render_single_tab(cost_mult)
