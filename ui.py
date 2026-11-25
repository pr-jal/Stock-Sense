import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

POPULAR = {
    "Reliance Industries (India)": "RELIANCE.NS", "Tata Consultancy Services - TCS (India)": "TCS.NS",
    "HDFC Bank (India)": "HDFCBANK.NS", "Infosys (India)": "INFY.NS", "ICICI Bank (India)": "ICICIBANK.NS",
    "State Bank of India - SBI (India)": "SBIN.NS", "Bharti Airtel (India)": "BHARTIARTL.NS",
    "ITC (India)": "ITC.NS", "Hindustan Unilever (India)": "HINDUNILVR.NS", "Larsen & Toubro (India)": "LT.NS",
    "Wipro (India)": "WIPRO.NS", "HCL Technologies (India)": "HCLTECH.NS", "Asian Paints (India)": "ASIANPAINT.NS",
    "Maruti Suzuki (India)": "MARUTI.NS", "Tata Steel (India)": "TATASTEEL.NS",
    "Bajaj Finance (India)": "BAJFINANCE.NS", "Adani Enterprises (India)": "ADANIENT.NS",
    "Sun Pharma (India)": "SUNPHARMA.NS", "Kotak Mahindra Bank (India)": "KOTAKBANK.NS",
    "NIFTY 50 index (India)": "^NSEI", "SENSEX index (India)": "^BSESN",
    "Apple (USA)": "AAPL", "Microsoft (USA)": "MSFT", "NVIDIA (USA)": "NVDA", "Alphabet / Google (USA)": "GOOGL",
    "Amazon (USA)": "AMZN", "Meta / Facebook (USA)": "META", "Tesla (USA)": "TSLA", "Netflix (USA)": "NFLX",
    "JPMorgan Chase (USA)": "JPM", "S&P 500 index (USA)": "^GSPC",
    "Other (type your own)": None,
}

FEATURE_NAMES = {
    "return_1d": "1-day return", "return_5d": "5-day return", "return_10d": "10-day return",
    "close_vs_sma10": "Price vs 10-day average", "close_vs_sma20": "Price vs 20-day average",
    "close_vs_sma50": "Price vs 50-day average", "sma10_vs_sma50": "10-day vs 50-day average",
    "rsi14": "RSI (14)", "macd": "MACD", "macd_signal": "MACD signal", "macd_hist": "MACD histogram",
    "volatility_10d": "10-day volatility", "volatility_20d": "20-day volatility",
    "volume_change": "Change in volume", "high_low_range": "High–low range",
}

CHART_HELP = {
    "Price": "Each candle is one day: green = closed higher than it opened, red = closed lower.<br>"
             "SMA 20 / SMA 50 = average closing price over the last 20 / 50 days. They show the trend.",
    "RSI (14)": "RSI (Relative Strength Index) scores recent momentum from 0 to 100.<br>"
                "Above 70 (top dashed line): the price rose a lot recently.<br>"
                "Below 30 (bottom dashed line): the price fell a lot recently.",
    "MACD": "MACD shows whether a trend is speeding up or slowing down.<br>"
            "MACD crossing above the Signal line suggests rising upward momentum;<br>"
            "crossing below suggests rising downward momentum.",
}

GREEN, RED, GREY = "#2e9e5b", "#d6455d", "#8a8f98"
BASELINE = "Baseline (majority class)"
MODEL_NAMES = [BASELINE, "Logistic Regression", "Random Forest", "Gradient Boosting"]


def css(dark: bool) -> str:
    bg = "rgb(23, 22, 20)" if dark else "rgb(255, 255, 255)"
    card_bg = "rgba(255,255,255,0.04)" if dark else "rgba(0,0,0,0.03)"
    border = "rgba(255,255,255,0.12)" if dark else "rgba(0,0,0,0.12)"
    return f"""<style>
[data-testid="stMainBlockContainer"], .block-container {{ padding-top: 1.5rem; }}
[data-testid="stSidebarHeader"] {{ height: 2rem; padding-top: 0.25rem; padding-bottom: 0; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0; }}
[data-testid="stSidebarUserContent"] h2 {{ padding: 0.25rem 0; font-size: 1.3rem; }}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}
[data-testid="stHeaderActionElements"] {{ display: none; }}
.st-key-theme_switch {{ position: fixed; top: 0.75rem; right: 1.5rem; z-index: 1000001; width: auto !important;
    background-color: {bg}; padding: 0.3rem 0.7rem; border-radius: 0.5rem; }}
.app-title {{ font-size: 2.4rem; font-weight: 700; line-height: 1.15; }}
.app-sub {{ font-size: 1rem; opacity: 0.75; margin: 0.3rem 0 0.6rem 0; }}
.badge {{ display: inline-block; padding: 0.15rem 0.6rem; margin: 0 0.35rem 0.35rem 0; font-size: 0.78rem;
    border: 1px solid {border}; border-radius: 999px; background: {card_bg}; }}
.card {{ border: 1px solid {border}; background: {card_bg}; border-radius: 0.6rem; padding: 0.7rem 0.9rem; height: 100%; }}
.card-label {{ font-size: 0.85rem; opacity: 0.8; }}
.card-value {{ font-size: 1.45rem; line-height: 1.25; padding-top: 0.25rem; overflow-wrap: break-word; }}
.card-sub {{ font-size: 0.8rem; opacity: 0.7; padding-top: 0.2rem; }}
.tip {{ display: inline-block; width: 1.05em; height: 1.05em; line-height: 1em; margin-left: 0.3em; font-size: 0.75em;
    border: 1px solid currentColor; border-radius: 50%; text-align: center; opacity: 0.6; cursor: help; }}
.flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin-bottom: 0.5rem; }}
.flow-box {{ background: #4c78a8; color: white; border-radius: 0.5rem; padding: 0.5rem 0.75rem; font-size: 0.9rem; line-height: 1.3; }}
.flow-box span {{ opacity: 0.85; font-size: 0.8rem; }}
.flow-arrow {{ font-size: 1.3rem; opacity: 0.6; }}
.howto {{ border-left: 3px solid #4c78a8; background: {card_bg}; padding: 0.5rem 0.8rem; border-radius: 0.3rem;
    font-size: 0.9rem; margin-bottom: 0.8rem; }}
.pred {{ border-radius: 0.7rem; padding: 1rem 1.2rem; color: white; }}
.pred-dir {{ font-size: 2rem; font-weight: 700; }}
@media (max-width: 640px) {{
    .app-title {{ font-size: 1.6rem; padding-right: 7rem; }}
    .st-key-theme_switch {{ top: 0.5rem; right: 0.5rem; }}
    .card-value {{ font-size: 1.2rem; }}
    .card {{ margin-bottom: 0.5rem; }}
    .pred-dir {{ font-size: 1.5rem; }}
}}
</style>"""

HEADER = (
    "<div class='app-title'>StockSense</div>"
    "<div class='app-sub'>Machine-learning stock direction prediction — will the next session close higher or lower? "
    "Educational project, not financial advice.</div>"
    + "".join(f"<span class='badge'>{b}</span>" for b in [
        "4 ML models", "15 engineered features", "Chronological + walk-forward validation",
        "Explainable predictions", "News sentiment (NLP)"]))


def how_to(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return f"<div class='howto'><b>How to read this:</b> {text}</div>"


def card(label, value, tip=None, sub=None) -> str:
    tip_html = f"<span class='tip' title='{tip}'>?</span>" if tip else ""
    sub_html = f"<div class='card-sub'>{sub}</div>" if sub else ""
    return (f"<div class='card'><div class='card-label'>{label}{tip_html}</div>"
            f"<div class='card-value'>{value}</div>{sub_html}</div>")


def prediction_box(up: bool, colour: str, *lines: str) -> str:
    body = "".join(f"<div>{line}</div>" for line in lines if line)
    return f"<div class='pred' style='background:{colour}'><div class='pred-dir'>{'▲ HIGHER' if up else '▼ LOWER'}</div>{body}</div>"


def flow(steps) -> str:
    boxes = "<span class='flow-arrow'>→</span>".join(
        f"<div class='flow-box'><b>{t}</b><br><span>{d}</span></div>" for t, d in steps)
    return f"<div class='flow'>{boxes}</div>"


def nice_table(t: pd.DataFrame) -> pd.DataFrame:
    t = t.round(3).rename(columns={"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1 score"})
    t.index.name = "Model"
    return t


def symbol(currency: str) -> str:
    return {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "")


def price(n, currency) -> str:
    return f"{symbol(currency)}{n:,.2f}"


def money(n, currency) -> str:
    if n is None or pd.isna(n):
        return "—"
    if currency == "INR":
        crore = n / 1e7
        return f"₹{crore / 1e5:,.2f} lakh crore" if crore >= 1e5 else f"₹{crore:,.0f} crore"
    sym = symbol(currency) or ((currency or "") + " ")
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(n) >= div:
            return f"{sym}{n / div:,.2f}{unit}"
    return f"{sym}{n:,.0f}"


def fmt_feature(f: str, v: float) -> str:
    if f == "rsi14":
        return f"{v:.0f}"
    if f.startswith("macd"):
        return f"{v:+.4f}"
    if f == "volume_change":
        return f"{np.expm1(v):+.0%}"
    if f.startswith("volatility") or f == "high_low_range":
        return f"{v:.2%}"
    return f"{v:+.2%}"


def pct(v, signed=False) -> str:
    return "" if pd.isna(v) else f"{v:+.1%}" if signed else f"{v:.1%}"


def bar_chart(x, series: dict, title: str, height=420, labels=True, y_max=None, **layout) -> go.Figure:
    fig = go.Figure([go.Bar(name=name, x=x, y=list(y), text=[pct(v) for v in y] if labels else None,
                            textposition="outside") for name, y in series.items()])
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey")
    fig.update_layout(barmode="group", height=height, title=title, yaxis_title="Accuracy",
                      yaxis_tickformat=".0%", xaxis_tickangle=0, **layout)
    if y_max:
        fig.update_yaxes(range=[0, y_max])
    return fig
