"""Streamlit dashboard for the Multi-Model LLM Tokenization Comparison System.

This is the presentation layer. The heavy HTML/CSS/chart building lives in
``components.py`` and ``styles.py``; the tokenization logic lives in
``orchestrator.py`` and the ``adapters/`` package.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from components import (
    build_rows,
    color_scale,
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

st.set_page_config(page_title="Tokenization Comparison", page_icon="🔤", layout="wide")
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
    "Chinese": "自然语言处理是人工智能的一个重要领域。",
}

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def render_sidebar() -> int:
    """Render the sidebar; return the selected cost multiplier."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section"><div class="title">🔤 TokenCompare</div>'
            '<div style="font-size:0.8rem;color:#64748b;">'
            "A multi-model LLM tokenization comparison dashboard.</div></div>",
            unsafe_allow_html=True,
        )

        # --- cache ---
        st.markdown('<div class="sidebar-section"><div class="title">Cache</div>', unsafe_allow_html=True)
        rate = orchestrator.hit_rate
        st.progress(rate, text=f"Cache hit rate: {rate:.0%}")
        st.markdown(
            f"💾 API calls saved: **{orchestrator.cache_hits}** &nbsp;·&nbsp; "
            f"Total lookups: {orchestrator.cache_hits + orchestrator.cache_misses}",
            unsafe_allow_html=True,
        )
        if st.button("Clear cache", width="stretch"):
            orchestrator.clear_cache()
            st.success("Cache cleared.")
        st.markdown("</div>", unsafe_allow_html=True)

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
                label = "Free (self-hosted)"
            elif price_per_million(m) > 0:
                label = f"${price_per_million(m):,.2f} / 1M tokens"
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
        f'<div class="hero-title">Multi-Model LLM Tokenization Comparison</div>'
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
        '<div style="font-size:1.6rem;">🔤</div>'
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
    text = render_input_section()
    if "last_results" not in st.session_state:
        render_empty_state()
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
# Batch tab
# --------------------------------------------------------------------------- #


def render_batch() -> None:
    st.markdown(
        "Upload a CSV with columns `category,text` (or load the sample file). "
        "Each row is tokenized by all three models."
    )
    sample_path = Path(__file__).resolve().parent / "samples" / "sample_texts.csv"

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    use_sample = st.button("Or load the bundled sample file")

    df_in = None
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
    elif use_sample and sample_path.exists():
        df_in = pd.read_csv(sample_path)

    if df_in is None:
        st.info("Upload a CSV or load the sample file to begin.")
        return

    if set(df_in.columns) != {"category", "text"}:
        st.error("The CSV must have exactly the columns 'category' and 'text'.")
        return

    rows = []
    progress = st.progress(0.0, text="Tokenizing…")
    total = len(df_in)
    for i, row in df_in.iterrows():
        cat, text = row["category"], row["text"]
        for r in orchestrator.compare(text):
            rows.append(
                {
                    "Category": cat,
                    "Model": display_name(r["model"]),
                    "Tokens": r["tokens"],
                    "Error": r["error"],
                }
            )
        progress.progress((i + 1) / total)
    progress.empty()

    combined = pd.DataFrame(rows)
    combined["Tokens"] = pd.to_numeric(combined["Tokens"], errors="coerce").astype("Int64")

    # --- summary ------------------------------------------------------------
    total_texts = len(df_in)
    successful = int(combined["Tokens"].notna().sum())
    failed = int(combined["Error"].notna().sum())
    avg = combined.groupby("Model")["Tokens"].mean().round(1).reset_index()
    avg.columns = ["Model", "Average tokens"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Input texts", total_texts)
    m2.metric("Successful comparisons", successful)
    m3.metric("Failed comparisons", failed)

    st.markdown("**Average tokens per model**")
    st.dataframe(avg, width="stretch", hide_index=True)

    st.subheader("Combined results")
    st.dataframe(combined[["Category", "Model", "Tokens", "Error"]], width="stretch", hide_index=True)

    chart_df = combined[combined["Tokens"].notna()]
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Category:N", sort=None, axis=alt.Axis(labelAngle=0, title="Category")),
            y=alt.Y("Tokens:Q", axis=alt.Axis(title="Tokens")),
            xOffset="Model:N",
            color=alt.Color("Model:N", scale=color_scale(), title="Model"),
            tooltip=["Category:N", "Model:N", "Tokens:Q"],
        )
        .properties(title="Tokens by category and model", height=320)
    )
    st.altair_chart(chart, width="stretch")

    st.download_button(
        "Download results (CSV)",
        combined.to_csv(index=False),
        file_name="batch_tokenization_results.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #

cost_mult = render_sidebar()
render_hero()

tab_single, tab_batch = st.tabs(["Single input", "Batch mode"])
with tab_single:
    render_single_tab(cost_mult)
with tab_batch:
    render_batch()
