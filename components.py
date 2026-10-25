"""Reusable rendering helpers for the dashboard.

These build the HTML metric cards, token chips, the auto-generated comparison
insights, the Altair charts and the data tables used by ``app.py``. Keeping the
HTML/CSS in ``styles.py`` and these components here keeps ``app.py`` readable.
"""

import html

import altair as alt
import pandas as pd
import streamlit as st

from cost import (
    display_name,
    estimate,
    get_model_info,
    has_price,
    is_open_source,
    price_per_million,
)
from orchestrator import MODEL_NAMES
from styles import chip_colours, model_color, tint

#: Static description of how each model tokenizes, shown on the cards.
TOKENIZER_METHOD = {
    "gpt-4o": "Byte-Pair Encoding (o200k_base) · local",
    "gemini-3.8-flash": "API · count only",
    "qwen2.5-0.5b": "Byte-Pair Encoding · local (Hugging Face)",
}

# --------------------------------------------------------------------------- #
# Cost formatting
# --------------------------------------------------------------------------- #


def fmt_money(value: float) -> str:
    """Format a USD amount, keeping enough precision for tiny values."""
    if value == 0:
        return "$0.00"
    if value >= 1000:
        return f"${value:,.2f}"
    if value >= 1:
        return f"${value:,.4f}"
    return "$" + f"{value:.8f}".rstrip("0").rstrip(".")


def format_cost(model: str, tokens: int, multiplier: int) -> str:
    """Human-readable cost for a model. Never shows a bare ``$0.00``."""
    if is_open_source(model):
        return "Free (self-hosted)"
    if not has_price(model):
        return "Price not set"
    return fmt_money(estimate(model, tokens) * multiplier)


def numeric_cost(model: str, tokens: int, multiplier: int):
    """Numeric cost for charting (None when the price is unset)."""
    if is_open_source(model):
        return 0.0
    if not has_price(model):
        return None
    return estimate(model, tokens) * multiplier


def short_source(url: str) -> str:
    """Shorten a pricing-source URL for display (drop protocol + trailing slash)."""
    if not url:
        return "—"
    return url.replace("https://", "").replace("http://", "").rstrip("/")


# --------------------------------------------------------------------------- #
# Altair colour scale (one colour per model)
# --------------------------------------------------------------------------- #


def color_scale() -> alt.Scale:
    """An Altair scale mapping each model's display name to its brand colour."""
    domain = [display_name(m) for m in MODEL_NAMES]
    rng = [model_color(m) for m in MODEL_NAMES]
    return alt.Scale(domain=domain, range=rng)


# --------------------------------------------------------------------------- #
# Result rows / winners / summary
# --------------------------------------------------------------------------- #


def build_rows(results: list[dict], text: str) -> list[dict]:
    """Enrich orchestrator results with display fields used by the UI."""
    n_chars = len(text)
    rows = []
    for r in results:
        rows.append(
            {
                "model": r["model"],
                "display_name": display_name(r["model"]),
                "tokens": r["tokens"],
                "chars_per_token": (n_chars / r["tokens"]) if r["tokens"] else None,
                "cached": r["cached"],
                "error": r["error"],
                "pieces": r.get("pieces"),
                "token_ids": r.get("token_ids"),
            }
        )
    return rows


def compute_winners(rows: list[dict]) -> dict:
    """Return ``{award: [model_keys]}``; ties keep every tied model.

    A model with an error is never eligible for an award.
    """
    valid = [r for r in rows if r["tokens"] and not r["error"]]
    winners = {}
    if valid:
        min_tokens = min(r["tokens"] for r in valid)
        winners["fewest_tokens"] = [r["model"] for r in valid if r["tokens"] == min_tokens]

        max_cpt = max(r["chars_per_token"] for r in valid)
        winners["most_efficient"] = [
            r["model"] for r in valid if abs(r["chars_per_token"] - max_cpt) < 1e-9
        ]

    priced = [r for r in valid if has_price(r["model"])]
    if priced:
        min_cost = min(estimate(r["model"], r["tokens"]) for r in priced)
        winners["cheapest"] = [
            r["model"]
            for r in priced
            if abs(estimate(r["model"], r["tokens"]) - min_cost) < 1e-9
        ]
    return winners


#: Shown under the insights to avoid misreading the density metric.
DENSITY_DISCLAIMER = (
    "Characters per token measures tokenization density — how much text each "
    "token covers. It is not a measure of model intelligence, response quality, "
    "or processing speed."
)


def comparison_insights(rows: list[dict]) -> list[str]:
    """Generate concise, data-driven observations from the comparison results.

    Every observation is computed from the actual results (never hardcoded),
    and ties are reported as ties rather than picking a single winner.
    """
    valid = [r for r in rows if r["tokens"] and not r["error"]]
    errored = [r for r in rows if r["error"]]
    if not valid:
        return []

    def names(rs):
        return ", ".join(r["display_name"] for r in rs)

    insights = []

    # --- token counts (fewest vs most, ties handled) -----------------------
    min_tokens = min(r["tokens"] for r in valid)
    max_tokens = max(r["tokens"] for r in valid)
    fewest = [r for r in valid if r["tokens"] == min_tokens]
    most = [r for r in valid if r["tokens"] == max_tokens]

    if len(valid) == 1:
        insights.append(
            f"Only {valid[0]['display_name']} returned a result, so no comparison could be made."
        )
    elif min_tokens == max_tokens:
        insights.append(f"All models used {min_tokens} tokens — a tie.")
    else:
        pct = (max_tokens - min_tokens) / min_tokens * 100
        insights.append(
            f"{names(fewest)} used the fewest tokens ({min_tokens}), while {names(most)} "
            f"used {max_tokens} tokens — {pct:.1f}% more than {fewest[0]['display_name']}."
        )

    # --- density (highest chars per token) ---------------------------------
    max_cpt = max(r["chars_per_token"] for r in valid)
    densest = [r for r in valid if abs(r["chars_per_token"] - max_cpt) < 1e-9]
    if len(densest) == 1:
        insights.append(
            f"{densest[0]['display_name']} has the highest characters-per-token ratio ({max_cpt:.2f})."
        )
    else:
        insights.append(
            f"{names(densest)} share the highest characters-per-token ratio ({max_cpt:.2f})."
        )

    # --- cost (lowest API price among paid models; open-source is free) ----
    api_priced = [r for r in valid if not is_open_source(r["model"]) and has_price(r["model"])]
    free_models = [r for r in valid if is_open_source(r["model"])]
    if api_priced:
        min_price = min(price_per_million(r["model"]) for r in api_priced)
        cheapest = [
            r for r in api_priced if abs(price_per_million(r["model"]) - min_price) < 1e-9
        ]
        if len(cheapest) == 1:
            insights.append(
                f"{cheapest[0]['display_name']} has the lowest API price "
                f"(${min_price:.2f} per 1M input tokens)."
            )
        else:
            insights.append(
                f"{names(cheapest)} tie for the lowest API price "
                f"(${min_price:.2f} per 1M input tokens)."
            )
    if free_models:
        insights.append(f"{names(free_models)} is open-source and runs for free (self-hosted).")

    # --- limitations --------------------------------------------------------
    if errored:
        insights.append(f"{names(errored)} could not be reached and was excluded from the comparison.")
    unpriced = [
        r for r in valid if not is_open_source(r["model"]) and not has_price(r["model"])
    ]
    if unpriced:
        insights.append(f"Cost comparison excludes {names(unpriced)} (price not configured).")

    return insights


def render_insights(rows: list[dict]) -> None:
    """Render the "Comparison Insights" section below the cards."""
    insights = comparison_insights(rows)
    if not insights:
        return
    items = "".join(f'<div class="insight">• {html.escape(s)}</div>' for s in insights)
    note = f'<div class="insight-note">{html.escape(DENSITY_DISCLAIMER)}</div>'
    st.markdown(
        f'<div class="insights-box">'
        f'<div class="insights-title">💡 Comparison Insights</div>'
        f'{items}{note}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Metric cards
# --------------------------------------------------------------------------- #


def _card(r: dict, winners: dict, multiplier: int) -> str:
    color = model_color(r["model"])
    name = r["display_name"]
    version = r["model"]

    if r["error"]:
        return (
            f'<div class="model-card" style="border-top:4px solid #ef4444;">'
            f'<div class="card-head">'
            f'<div class="card-title" style="color:#ef4444;">{name}</div>'
            f'<div class="card-version">{version}</div></div>'
            f'<div class="card-error">⚠️ Could not reach this model</div>'
            f'<div class="card-meta">{html.escape(r["error"])}</div></div>'
        )

    cpt = f"{r['chars_per_token']:.2f}" if r["chars_per_token"] else "—"
    cost = format_cost(r["model"], r["tokens"], multiplier)
    method = TOKENIZER_METHOD.get(r["model"], "—")
    info = get_model_info(r["model"])
    src = short_source(info.get("price_source", ""))
    date = info.get("price_date", "")
    src_line = f'<div class="cost-source">Source: {src}' + (f" · {date}" if date else "") + "</div>"

    pill = (
        '<span class="pill pill-cached">⚡ Cached</span>'
        if r["cached"]
        else '<span class="pill pill-live">🌐 Live API</span>'
    )
    badges = ""
    for key, label in (
        ("fewest_tokens", "🏆 Fewest tokens"),
        ("cheapest", "💰 Cheapest"),
        ("most_efficient", "⚡ Most efficient"),
    ):
        if r["model"] in winners.get(key, []):
            badges += f'<span class="winner-badge">{label}</span>'

    return (
        f'<div class="model-card" style="border-top:4px solid {color};">'
        f'<div class="card-head">'
        f'<div class="card-title" style="color:{color};">{name}</div>'
        f'<div class="card-version">{version}</div></div>'
        f'<div class="card-tokens">{r["tokens"]:,}<span class="card-unit"> tokens</span></div>'
        f'<div class="card-meta">'
        f'<div class="meta-row"><span class="meta-label">Chars / token</span><span class="meta-value">{cpt}</span></div>'
        f'<div class="meta-row"><span class="meta-label">Est. cost</span><span class="meta-value">{cost}</span></div>'
        f'<div class="meta-row"><span class="meta-label">Method</span><span class="meta-value">{method}</span></div>'
        f'{src_line}</div>'
        f'<div class="card-foot">{pill}{badges}</div></div>'
    )


def render_cards(rows: list[dict], multiplier: int) -> dict:
    """Render the model metric cards; return the winners dict."""
    winners = compute_winners(rows)
    cards = "".join(_card(r, winners, multiplier) for r in rows)
    st.markdown(f'<div class="card-row">{cards}</div>', unsafe_allow_html=True)
    return winners


# --------------------------------------------------------------------------- #
# Token breakdown (chips)
# --------------------------------------------------------------------------- #


def _token_display(piece: str) -> str:
    if piece == "":
        return "␀"
    return piece.replace("\n", "⏎").replace("\t", "⇥").replace(" ", "␣")


def render_chips_html(r: dict) -> str:
    """Build the HTML for one model's token chips (with hover tooltips)."""
    pieces = r["pieces"]
    ids = r.get("token_ids") or [None] * len(pieces)
    light, lighter = chip_colours(r["model"])
    border = tint(model_color(r["model"]), 0.55)
    chips = []
    for i, (piece, tid) in enumerate(zip(pieces, ids)):
        bg = light if i % 2 == 0 else lighter
        tip = f"token #{i} · id {tid}" if tid is not None else f"token #{i}"
        chips.append(
            f'<span class="token-chip" style="background:{bg};border-color:{border};" '
            f'title="{html.escape(tip)}">{html.escape(_token_display(piece))}</span>'
        )
    return f'<div class="chip-wrap">{"".join(chips)}</div>'


def render_breakdown(rows: list[dict]) -> None:
    """Show token chips for each model; clearly flag count-only models."""
    models = [r for r in rows if not r["error"]]
    if not models:
        st.info("No token breakdown available for this text.")
        return
    cols = st.columns(len(models))
    for col, r in zip(cols, models):
        color = model_color(r["model"])
        with col:
            cpt = f"{r['chars_per_token']:.2f}" if r["chars_per_token"] else "—"
            st.markdown(
                f'<div style="font-weight:700;color:{color};">{r["display_name"]}</div>'
                f'<div style="font-size:0.8rem;color:#64748b;margin-bottom:0.25rem;">'
                f'{r["tokens"]:,} tokens · {cpt} chars/token</div>',
                unsafe_allow_html=True,
            )
            if r["pieces"]:
                st.markdown(render_chips_html(r), unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="no-pieces">Token pieces are not available — {r["display_name"]} uses a '
                    f'proprietary server-side tokenizer, so the API exposes only a total token count.</div>',
                    unsafe_allow_html=True,
                )
    st.caption(
        "Each chip is one token; whitespace is shown as ␣ (space), ⏎ (newline) and ⇥ (tab). "
        "Hover a chip to see its position and vocabulary ID."
    )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def chart_df(rows: list[dict], multiplier: int) -> pd.DataFrame:
    valid = [r for r in rows if r["tokens"] and not r["error"]]
    min_tokens = min((r["tokens"] for r in valid), default=None)
    data = []
    for r in valid:
        rel = ((r["tokens"] - min_tokens) / min_tokens * 100) if min_tokens else 0.0
        data.append(
            {
                "Model": r["display_name"],
                "Tokens": r["tokens"],
                "Cost": numeric_cost(r["model"], r["tokens"], multiplier),
                "Chars per token": r["chars_per_token"],
                "Relative (%)": rel,
            }
        )
    return pd.DataFrame(data)


def _bar(df: pd.DataFrame, y: str, title: str, ytitle: str, fmt: str) -> alt.Chart:
    base = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Model:N", sort=None, axis=alt.Axis(labelAngle=0, title=None, labelFontSize=12)),
            y=alt.Y(f"{y}:Q", axis=alt.Axis(title=ytitle, grid=True, titleFontSize=12)),
            color=alt.Color("Model:N", scale=color_scale(), legend=None),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip(f"{y}:Q", title=ytitle, format=fmt),
            ],
        )
    )
    labels = base.mark_text(align="center", dy=-10, fontSize=12, fontWeight="bold").encode(
        text=alt.Text(f"{y}:Q", format=fmt)
    )
    return (base + labels).properties(
        title=alt.Title(title, fontSize=14, fontWeight=600), height=300
    )


def render_charts(rows: list[dict], multiplier: int) -> None:
    df = chart_df(rows, multiplier)
    cost_df = df[df["Cost"].notna()]

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(_bar(df, "Tokens", "Tokens per model", "Tokens", ",d"), width="stretch")
    with c2:
        st.altair_chart(
            _bar(df, "Relative (%)", "Relative Token Usage (%)", "Difference vs fewest (%)", ".1f"),
            width="stretch",
        )

    c3, c4 = st.columns(2)
    with c3:
        st.altair_chart(
            _bar(cost_df, "Cost", "Estimated cost per model", "Cost (USD)", ".8~f"),
            width="stretch",
        )
    with c4:
        st.altair_chart(
            _bar(df, "Chars per token", "Efficiency per model", "Chars per token", ".2f"),
            width="stretch",
        )


# --------------------------------------------------------------------------- #
# Cost simulator
# --------------------------------------------------------------------------- #


def render_cost_simulator(rows: list[dict]) -> None:
    """A simple cost simulator: cost-per-request × number of requests."""
    models = [r for r in rows if r["tokens"] is not None and not r["error"]]
    if not models:
        return

    st.markdown('<div class="section-title">💸 Cost Simulator</div>', unsafe_allow_html=True)
    n = st.select_slider(
        "Number of requests",
        options=[1_000, 10_000, 100_000, 1_000_000],
        value=10_000,
        format_func=lambda x: f"{x:,}",
    )

    data = []
    for r in models:
        model = r["model"]
        if is_open_source(model):
            per = "N/A (Local Tokenizer)"
            total = "N/A (No API Pricing)"
        elif not has_price(model):
            per = "—"
            total = "Price unavailable"
        else:
            per_cost = estimate(model, r["tokens"])
            per = fmt_money(per_cost)
            total = fmt_money(per_cost * n)
        data.append(
            {
                "Model": r["display_name"],
                "Cost per request": per,
                f"Total cost ({n:,} requests)": total,
            }
        )

    st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)
    st.caption(
        "Assumes the same input text is processed repeatedly and excludes output-token "
        "charges. Models shown as N/A (Local Tokenizer) tokenize locally with no configured "
        "commercial API charge; this does not imply zero operating cost — self-hosting "
        "infrastructure and model-inference costs are excluded."
    )


# --------------------------------------------------------------------------- #
# Data table
# --------------------------------------------------------------------------- #


def table_display(rows: list[dict], multiplier: int) -> pd.DataFrame:
    """Formatted table for on-screen display."""
    data = []
    for r in rows:
        data.append(
            {
                "Model": r["display_name"],
                "Tokens": f"{r['tokens']:,}" if r["tokens"] is not None else "—",
                "Characters per token": (
                    f"{r['chars_per_token']:.2f}" if r["chars_per_token"] else "—"
                ),
                "Estimated cost": "—"
                if r["error"]
                else format_cost(r["model"], r["tokens"], multiplier),
                "Price source": "—"
                if r["error"]
                else short_source(get_model_info(r["model"]).get("price_source", "")),
                "Status": "⚠️ Error" if r["error"] else ("⚡ Cached" if r["cached"] else "🌐 Live"),
            }
        )
    return pd.DataFrame(data)


def table_raw(rows: list[dict], multiplier: int) -> pd.DataFrame:
    """Raw (numeric) table for CSV download."""
    data = []
    for r in rows:
        data.append(
            {
                "model": r["model"],
                "display_name": r["display_name"],
                "tokens": r["tokens"],
                "chars_per_token": round(r["chars_per_token"], 4) if r["chars_per_token"] else None,
                "cost_usd": numeric_cost(r["model"], r["tokens"], multiplier) if not r["error"] else None,
                "cached": r["cached"],
                "error": r["error"],
            }
        )
    return pd.DataFrame(data)


def render_table(rows: list[dict], multiplier: int) -> None:
    st.dataframe(table_display(rows, multiplier), width="stretch", hide_index=True)
    st.download_button(
        "Download CSV",
        table_raw(rows, multiplier).to_csv(index=False),
        file_name="tokenization_results.csv",
        mime="text/csv",
    )
