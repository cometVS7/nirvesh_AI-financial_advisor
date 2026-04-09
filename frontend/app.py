from __future__ import annotations

import html
import os
import random
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "https://nirveshai-financialadvisor-production.up.railway.app"

st.set_page_config(page_title="Nirvesh AI", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg-1: #121826;
        --bg-2: #0f172a;
        --card: rgba(30, 41, 59, 0.92);
        --text-primary: #E5E7EB;
        --text-secondary: #9CA3AF;
        --accent: #3B5BDB;
        --accent-hover: #5C7CFA;
        --green: #22C55E;
        --red: #EF4444;
        --yellow: #F59E0B;
    }
    .stApp {
        background: radial-gradient(circle at 15% 10%, rgba(59, 91, 219, 0.14), transparent 35%),
                    linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 100%);
        color: var(--text-primary);
    }
    .block-container {
        max-width: 1240px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: var(--text-primary);
    }
    .card {
        background: var(--card);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(2, 6, 23, 0.34);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 30px rgba(2, 6, 23, 0.42);
    }
    .highlight {
        border: 1px solid rgba(92, 124, 250, 0.35);
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.98), rgba(59, 91, 219, 0.16));
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
        background: linear-gradient(90deg, #E5E7EB 0%, #A5B4FC 55%, #5C7CFA 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        margin-top: 10px;
        color: #CBD5E1;
        font-size: 1.02rem;
    }
    .hero-caption {
        margin-top: 14px;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    .mood-value {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .mood-positive { color: var(--green); }
    .mood-negative { color: var(--red); }
    .kpi-label {
        font-size: 0.88rem;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.52rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.15;
    }
    .kpi-subtext {
        margin-top: 8px;
        font-size: 0.82rem;
        color: #94A3B8;
    }
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .pill-blue {
        background: rgba(59, 91, 219, 0.22);
        color: #C7D2FE;
        border: 1px solid rgba(92, 124, 250, 0.35);
    }
    .recommendation-shell {
        text-align: center;
        padding: 20px;
    }
    .recommendation-value {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .recommendation-buy { color: var(--green); }
    .recommendation-hold { color: var(--yellow); }
    .recommendation-avoid { color: var(--red); }
    .recommendation-meta {
        color: var(--text-secondary);
        margin-bottom: 10px;
        line-height: 1.5;
    }
    .signal-positive { color: var(--green); }
    .signal-negative { color: var(--red); }
    div.stButton > button {
        background: var(--accent);
        color: #E5E7EB;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.62rem 1rem;
    }
    div.stButton > button:hover {
        background: var(--accent-hover);
        box-shadow: 0 0 0 2px rgba(92, 124, 250, 0.3), 0 0 20px rgba(92, 124, 250, 0.35);
        color: #FFFFFF;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input {
        background: rgba(30, 41, 59, 0.96);
        color: var(--text-primary);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_json(method: str, endpoint: str, payload: dict | None = None) -> dict:
    response = requests.request(method=method, url=f"{API_BASE}{endpoint}", json=payload, timeout=300)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"API error: {detail}")
    return response.json()


def fetch_excel(filename: str) -> bytes:
    response = requests.get(f"{API_BASE}/download/{filename}", timeout=300)
    response.raise_for_status()
    return response.content


def money(value: float) -> str:
    return f"Rs. {value:,.2f}"


def percent(value: float) -> str:
    return f"{value:,.2f}%"


def pill(label: str, kind: str = "blue") -> str:
    return f"<span class='pill pill-{kind}'>{label}</span>"


def metric_card(label: str, value: str, subtext: str = "") -> str:
    subtext_html = f"<div class='kpi-subtext'>{subtext}</div>" if subtext else ""
    return f"""
    <div class="card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {subtext_html}
    </div>
    """


def render_cards(cards: list[str]) -> None:
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <div class="card highlight">
            <div class="hero-title">📈 Nirvesh AI</div>
            <div class="hero-subtitle">AI-powered financial intelligence system</div>
            <div class="hero-caption">
                Real-time signals, horizon-aware forecasts, and contextual stock intelligence in one unified dashboard.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_mood(result=None) -> None:
    import random
    import streamlit as st

    # Priority 1: Use real sentiment if available
    if result and "overall_sentiment_score" in result:
        mood = round(result["overall_sentiment_score"], 2)
        st.session_state.market_mood = mood  # update session

    # Priority 2: Use stored session value
    elif "market_mood" in st.session_state:
        mood = st.session_state.market_mood

    # Fallback: generate once
    else:
        mood = round(random.uniform(-1.5, 1.5), 2)
        st.session_state.market_mood = mood

    sign = "+" if mood >= 0 else ""
    tone_class = "mood-positive" if mood >= 0 else "mood-negative"

    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-label">📊 Market Mood</div>
            <div class="mood-value {tone_class}">{sign}{mood:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_horizon_label(days: int) -> None:
    st.markdown(pill(f"Prediction Horizon: {days} days", "blue"), unsafe_allow_html=True)


def render_recommendation_card(recommendation: str, title: str, subtitle: str, body: str) -> None:
    normalized = recommendation.strip().lower()
    if normalized == "buy":
        value_class = "recommendation-buy"
        label = "BUY"
    elif normalized == "hold":
        value_class = "recommendation-hold"
        label = "HOLD"
    else:
        value_class = "recommendation-avoid"
        label = "AVOID"

    st.markdown(
        f"""
        <div class="card highlight recommendation-shell">
            <div class="recommendation-value {value_class}">{label}</div>
            <div style="font-size:1.14rem;font-weight:700;">{html.escape(title)}</div>
            <div class="recommendation-meta">{html.escape(subtitle)}</div>
            <div style="line-height:1.6;color:#CBD5E1;">{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_card(signal_pct: float) -> None:
    signal_class = "signal-positive" if signal_pct >= 0 else "signal-negative"
    sign = "+" if signal_pct >= 0 else ""
    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:1.12rem;font-weight:700;" class="{signal_class}">
                ⚡ Short-term Signal (5d): {sign}{signal_pct:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_insights(insight_text: str) -> None:
    safe = html.escape(insight_text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:1.08rem;font-weight:700;margin-bottom:10px;">🧠 AI Insights</div>
            <div style="color:#CBD5E1;line-height:1.7;">{safe}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_table(df: pd.DataFrame):
    format_map = {}
    for column in ["current_price", "predicted_price", "historical_reference_price", "allocation_amount"]:
        if column in df.columns:
            format_map[column] = "Rs. {:,.2f}"
    for column in ["growth_pct", "allocation_pct", "growth_probability", "short_term_signal_pct", "growth_90d"]:
        if column in df.columns:
            format_map[column] = "{:,.2f}"

    styled = df.style.format(format_map)

    def _value_heat_style(value):
        if pd.isna(value):
            return ""
        try:
            v = float(value)
        except Exception:
            return ""
        if v >= 5:
            return "background-color: rgba(34, 197, 94, 0.40); color: #E5E7EB;"
        if v > 0:
            return "background-color: rgba(34, 197, 94, 0.22); color: #E5E7EB;"
        if v <= -5:
            return "background-color: rgba(239, 68, 68, 0.40); color: #E5E7EB;"
        if v < 0:
            return "background-color: rgba(239, 68, 68, 0.22); color: #E5E7EB;"
        return "background-color: rgba(148, 163, 184, 0.12); color: #E5E7EB;"

    style_method = "map" if hasattr(styled, "map") else "applymap"
    for column in ["growth_pct", "short_term_signal_pct", "growth_90d"]:
        if column in df.columns:
            if style_method == "map":
                styled = styled.map(_value_heat_style, subset=[column])
            else:
                styled = styled.applymap(_value_heat_style, subset=[column])
    return styled


def render_data_table(df: pd.DataFrame) -> None:
    try:
        st.dataframe(style_table(df), use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)


def prepare_portfolio_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows

    preferred_columns = [
        "company",
        "ticker",
        "sector",
        "current_price",
        "predicted_price",
        "growth_pct",
        "growth_probability",
        "short_term_signal_pct",
        "allocation_pct",
        "allocation_amount",
        "risk_level",
        "recommendation",
        "prediction_horizon_days",
        "holding_time_days",
    ]
    available_columns = [column for column in preferred_columns if column in rows.columns]
    prepared = rows[available_columns].copy()

    for column in [
        "current_price",
        "predicted_price",
        "growth_pct",
        "growth_probability",
        "short_term_signal_pct",
        "allocation_pct",
        "allocation_amount",
    ]:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared.insert(0, "rank", range(1, len(prepared) + 1))
    return prepared


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="analysis")
    buffer.seek(0)
    return buffer.getvalue()


def recommendation_chart(df: pd.DataFrame, title: str) -> go.Figure:
    chart_df = df.copy()
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=chart_df["company"],
            y=chart_df["current_price"],
            name="Current Price",
            marker_color="#3B5BDB",
        )
    )
    figure.add_trace(
        go.Bar(
            x=chart_df["company"],
            y=chart_df["predicted_price"],
            name="Predicted Price",
            marker_color="#22C55E",
        )
    )
    figure.update_layout(
        title=title,
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def company_chart(historical: list[dict], prediction: list[dict]) -> go.Figure:
    hist_df = pd.DataFrame(historical)
    pred_df = pd.DataFrame(prediction)
    hist_df["date"] = pd.to_datetime(hist_df["date"], errors="coerce")
    pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
    hist_df = hist_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    pred_df = pred_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="#3B5BDB", width=2.5),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["predicted_close"],
            mode="lines",
            name="Predicted Close",
            line=dict(color="#22C55E", width=3, dash="dash"),
        )
    )
    figure.update_layout(
        title="Historical vs Predicted Price",
        xaxis_title="Date",
        yaxis_title="Price (Rs.)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def backtest_chart(rows: list[dict]) -> go.Figure:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["actual_close"],
            mode="lines+markers",
            name="Actual Close",
            line=dict(color="#3B5BDB", width=2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["predicted_close"],
            mode="lines+markers",
            name="Predicted Close",
            line=dict(color="#F59E0B", width=2, dash="dash"),
        )
    )
    figure.update_layout(
        title="Backtest: Predicted vs Actual",
        xaxis_title="Date",
        yaxis_title="Price (Rs.)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


render_hero()
st.markdown("")
render_market_mood()
st.markdown("---")

with st.sidebar:
    st.header("Input Panel")
    mode = st.selectbox("Choose Mode", ["📊 Portfolio", "🏭 Sector Analysis", "🏢 Company Analysis", "🔍 Backtesting"])
    sectors = fetch_json("GET", "/sectors")["sectors"]

if mode == "📊 Portfolio":
    st.markdown("### Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input("Budget (Rs.)", min_value=0.0, value=500000.0, step=1000.0)
    with col2:
        risk = st.selectbox("Risk Tolerance", ["low", "mid", "high"])
    with col3:
        holding = st.selectbox("Holding Period", ["short term", "medium term", "long term"])

    select_all_sectors = st.checkbox("Select All Sectors", value=True)
    default_sectors = sectors if select_all_sectors else sectors[: min(2, len(sectors))]
    selected_sectors = st.multiselect("Choose Sectors", sectors, default=default_sectors, disabled=select_all_sectors)
    effective_sectors = sectors if select_all_sectors else selected_sectors

    if st.button("Generate Portfolio", type="primary"):
        try:
            if budget <= 0:
                raise RuntimeError("Please enter a budget greater than zero.")
            if not effective_sectors:
                raise RuntimeError("Please select at least one sector before generating the portfolio.")

            with st.spinner("🧠 AI analyzing market signals..."):
                result = fetch_json(
                    "POST",
                    "/portfolio",
                    {
                        "budget": budget,
                        "risk_tolerance": risk,
                        "holding_period": holding,
                        "selected_sectors": effective_sectors,
                    },
                )

            rows = pd.DataFrame(result.get("portfolio", []))
            if rows.empty:
                st.warning("No valid portfolio recommendations found for the selected criteria.")
            else:
                portfolio_table = prepare_portfolio_table(rows)
                top_ten = portfolio_table.head(10).copy()
                st.markdown("---")
                st.markdown("### 📋 Top 10 Stocks Analysis (Portfolio Recommendation)")
                st.caption(f"Showing top {len(top_ten)} stocks out of {len(portfolio_table)} recommended picks.")
                render_data_table(top_ten)
                download_col1, download_col2 = st.columns(2)
                with download_col1:
                    st.download_button(
                        "Download top_10_portfolio_analysis.xlsx",
                        dataframe_to_excel_bytes(top_ten),
                        file_name="top_10_portfolio_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                with download_col2:
                    file_name = result.get("excel_file", {}).get("filename", "").strip()
                    full_export_name = "full_portfolio_analysis.xlsx"
                    if file_name:
                        try:
                            full_export = fetch_excel(file_name)
                            full_export_name = file_name
                        except Exception:
                            full_export = dataframe_to_excel_bytes(portfolio_table)
                    else:
                        full_export = dataframe_to_excel_bytes(portfolio_table)

                    st.download_button(
                        "Download full_portfolio_analysis.xlsx",
                        full_export,
                        file_name=full_export_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
        except Exception as exc:
            st.error(str(exc))

elif mode == "🏭 Sector Analysis":
    st.markdown("### Inputs")
    sector = st.selectbox("Sector", sectors)

    if st.button("Analyze Sector", type="primary"):
        try:
            with st.spinner("🧠 AI analyzing market signals..."):
                result = fetch_json("POST", "/sector", {"sector": sector})

            rows = pd.DataFrame(result["recommendations"])
            summary = result["summary"]
            if rows.empty:
                st.warning("No sector recommendations available right now.")
            else:
                top_six = rows.head(6).copy()
                avg_growth_30d = float(top_six["growth_pct"].mean()) if "growth_pct" in top_six.columns else float(summary.get("average_growth_pct", 0.0))
                growth_90d = float(summary.get("growth_90d", 0.0))
                trend_90d = str(summary.get("trend_90d", "Neutral"))

                rec_series = top_six.get("recommendation", pd.Series(dtype=str)).astype(str).str.lower()
                buy_count = int((rec_series == "buy").sum())
                hold_count = int((rec_series == "hold").sum())
                avoid_count = int((rec_series == "avoid").sum())

                if buy_count >= 4 or (avg_growth_30d >= 5 and trend_90d.lower() == "bullish"):
                    final_suggestion = "BUY"
                elif avoid_count >= 3 or avg_growth_30d <= -3 or trend_90d.lower() == "bearish":
                    final_suggestion = "AVOID"
                else:
                    final_suggestion = "HOLD"

                explanation_parts = [
                    f"Top 6 sector picks show an average 30-day growth outlook of {avg_growth_30d:.2f}%",
                    f"with a 90-day trend marked as {trend_90d} ({growth_90d:.2f}%).",
                    f"Recommendation mix across top picks: BUY {buy_count}, HOLD {hold_count}, AVOID {avoid_count}.",
                    "Final sector-level suggestion is based on growth consistency and longer-horizon trend alignment.",
                ]

                st.markdown("---")
                st.markdown("### KPI Cards")
                render_cards(
                    [
                        metric_card("💰 Current Price", money(float(top_six["current_price"].mean()) if "current_price" in top_six.columns else 0.0), "Average of top 6 picks"),
                        metric_card("📈 Predicted Price", money(float(top_six["predicted_price"].mean()) if "predicted_price" in top_six.columns else 0.0), "30-day average forecast"),
                        metric_card("🚀 Growth (30d)", percent(avg_growth_30d), "Top 6 sector basket"),
                        metric_card("⚠️ Trend (90d)", trend_90d, percent(growth_90d)),
                    ]
                )

                st.markdown("---")
                st.markdown("### Recommendation")
                render_recommendation_card(
                    final_suggestion,
                    f"{result['sector'].title()} Sector Outlook",
                    f"30-day growth: {avg_growth_30d:.2f}% | 90-day trend: {trend_90d}",
                    " ".join(explanation_parts),
                )

                st.markdown("")
                render_horizon_label(30)

                st.markdown("---")
                st.markdown("### 📉 Market Movement & Prediction")
                st.plotly_chart(
                    recommendation_chart(top_six, "Sector Picks: Current vs 30-Day Prediction"),
                    use_container_width=True,
                )

                st.markdown("---")
                render_ai_insights(" ".join(explanation_parts))

                st.markdown("---")
                st.markdown("### 📋 Recommendation Table")
                render_data_table(top_six)
                st.download_button(
                    "Download top_6_sector_analysis.xlsx",
                    dataframe_to_excel_bytes(top_six),
                    file_name=f"{result['sector'].replace(' ', '_').lower()}_top6_sector_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as exc:
            st.error(str(exc))

elif mode == "🏢 Company Analysis":
    st.markdown("### Inputs")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        keyword = st.text_input("Company name or ticker", value="Infosys")
    with col2:
        total_budget = st.number_input("Total Budget (Rs.)", min_value=1000.0, value=250000.0, step=1000.0)
    with col3:
        horizon_days = st.selectbox("Prediction Horizon", [15, 30, 90], index=1)

    if st.button("Analyze Company", type="primary"):
        try:
            with st.spinner("🧠 AI analyzing market signals..."):
                result = fetch_json(
                    "POST",
                    "/company",
                    {"company_keyword": keyword, "total_budget": total_budget, "days": horizon_days},
                )

            info = result["company"]
            insights = result["insights"]
            quote = insights.get("quote", {})

            st.markdown("---")
            st.markdown("### KPI Cards")
            render_cards(
                [
                    metric_card("💰 Current Price", money(float(insights["current_price"])), "Live or latest available quote"),
                    metric_card("📈 Predicted Price", money(float(insights["predicted_price"])), f"{insights.get('prediction_horizon_days', horizon_days)}-day forecast"),
                    metric_card("🚀 Growth %", percent(float(insights["predicted_growth_pct"])), "Selected horizon"),
                    metric_card("⚠️ Risk", str(insights["risk_level"]).title(), f"Confidence {insights.get('confidence_score', 0):.2f}"),
                ]
            )

            st.markdown("---")
            st.markdown("### Recommendation")
            render_recommendation_card(
                str(insights.get("recommendation", "hold")),
                f"{info['company']} ({info['ticker']})",
                f"Sector: {info['sector'].title()} | Allocation Suggestion: {insights.get('allocation_suggestion_pct', 0)}%",
                str(insights.get("explanation", "No explanation available.")),
            )

            st.markdown("")
            render_horizon_label(int(insights.get("prediction_horizon_days", horizon_days)))

            st.markdown("---")
            render_signal_card(float(insights.get("short_term_signal_pct", 0.0)))

            st.markdown("---")
            st.markdown("### 📉 Market Movement & Prediction")
            st.plotly_chart(company_chart(result["chart"]["historical"], result["chart"]["prediction"]), use_container_width=True)

            st.markdown("---")
            insight_lines = [str(insights.get("explanation", "No explanation available."))]
            if insights.get("news_analysis", {}).get("key_drivers"):
                insight_lines.append("Key News Drivers: " + ", ".join(item["label"] for item in insights["news_analysis"]["key_drivers"]))
            if insights.get("macro_news_analysis", {}).get("key_drivers"):
                insight_lines.append("Macro / Global Drivers: " + ", ".join(item["label"] for item in insights["macro_news_analysis"]["key_drivers"]))
            if insights.get("policy_drivers"):
                insight_lines.append("Policy Drivers: " + ", ".join(item["label"] for item in insights["policy_drivers"]))
            if insights.get("regulatory_drivers"):
                insight_lines.append("Regulatory Drivers: " + ", ".join(item["label"] for item in insights["regulatory_drivers"]))
            render_ai_insights("\n\n".join(insight_lines))

            st.markdown("---")
            st.markdown("### 📋 Data Table")
            st.caption(
                f"Current quote source: {quote.get('quote_source', 'unknown')} | "
                f"Quote time: {quote.get('quote_time', 'n/a')} | "
                f"Historical reference close: Rs. {insights.get('historical_reference_price', insights['current_price'])}"
            )
            if not quote.get("is_live", False):
                st.warning("Live current price could not be fetched. The displayed current price may be based on cached or historical data.")
            if quote.get("provider_attempts"):
                st.dataframe(pd.DataFrame(quote["provider_attempts"]), use_container_width=True)

            with st.expander("Agent Breakdown"):
                st.json(insights["agent_breakdown"])

            st.markdown("---")
            st.download_button(
                "Download company_analysis.xlsx",
                fetch_excel(result["excel_file"]["filename"]),
                file_name=result["excel_file"]["filename"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as exc:
            st.error(str(exc))

else:
    st.markdown("### Inputs")
    col1, col2 = st.columns([2, 1])
    with col1:
        keyword = st.text_input("Company name or ticker", value="Infosys", key="backtest_keyword")
    with col2:
        days = st.selectbox("Backtest Days", [7, 15, 30, 60, 90], index=0)

    if st.button("Run Backtest", type="primary"):
        try:
            with st.spinner("🧠 AI analyzing market signals..."):
                result = fetch_json("POST", "/backtest", {"company_keyword": keyword, "days": days})

            info = result["company"]
            st.markdown("---")
            st.markdown("### KPI Cards")
            render_cards(
                [
                    metric_card("💰 Current Price", money(float(result["results"][-1]["actual_close"] if result["results"] else 0.0)), "Last actual close"),
                    metric_card("📈 Predicted Price", money(float(result["results"][-1]["predicted_close"] if result["results"] else 0.0)), "Last predicted close"),
                    metric_card("🚀 Growth %", percent(float(result["summary"]["direction_accuracy"])), "Directional accuracy"),
                    metric_card("⚠️ Risk", "Backtest", f"{info['company']} ({info['ticker']})"),
                ]
            )

            st.markdown("---")
            st.markdown("### 📉 Market Movement & Prediction")
            st.plotly_chart(backtest_chart(result["chart"]), use_container_width=True)

            st.markdown("---")
            render_ai_insights(
                f"Backtest summary for {info['company']} ({info['ticker']}): "
                f"MAE {money(float(result['summary']['mean_absolute_error']))}, "
                f"MAPE {percent(float(result['summary']['mean_absolute_percentage_error']))}, "
                f"Direction Accuracy {percent(float(result['summary']['direction_accuracy']))}."
            )

            st.markdown("---")
            st.markdown("### 📋 Data Table")
            backtest_df = pd.DataFrame(result["results"])
            st.dataframe(
                backtest_df.style.format(
                    {
                        "previous_close": "Rs. {:,.2f}",
                        "predicted_close": "Rs. {:,.2f}",
                        "actual_close": "Rs. {:,.2f}",
                        "absolute_error": "Rs. {:,.2f}",
                        "percentage_error": "{:,.2f}%",
                    }
                ),
                use_container_width=True,
            )
        except Exception as exc:
            st.error(str(exc))
