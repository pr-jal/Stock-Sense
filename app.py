import datetime as dt
import importlib
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data_loader, features, model, news, ui
for _m in (data_loader, features, model, news, ui):
    importlib.reload(_m)

from data_loader import data_summary, get_company_info, get_prices, load_csv, log_prediction, prediction_history
from features import add_indicators, build_features, model_table
from model import backtest, explain_prediction, predict_next_day, run_pipeline, walk_forward
from news import get_news_sentiment
from ui import (BASELINE, CHART_HELP, FEATURE_NAMES, GREEN, GREY, MODEL_NAMES, POPULAR, RED,
                bar_chart, card, fmt_feature, how_to, money, nice_table, pct, prediction_box, price)

md = lambda html: st.markdown(html, unsafe_allow_html=True)

st.set_page_config(page_title="StockSense", layout="wide")
dark_now = st.session_state.get("dark_mode", True)
md(ui.css(dark_now))

with st.container(key="theme_switch"):
    dark = st.toggle("Dark mode", value=dark_now, key="dark_mode")
if st._config.get_option("theme.base") != ("dark" if dark else "light"):
    st._config.set_option("theme.base", "dark" if dark else "light")
    st.rerun()

md(ui.HEADER)

with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Yahoo Finance", "Upload CSV"],
                      help="Where the price data comes from: downloaded from Yahoo Finance, or a CSV file you upload.")
    start, end, refresh, upload = dt.date(2015, 1, 1), dt.date.today(), False, None
    if source == "Yahoo Finance":
        choice = st.selectbox("Stock", list(POPULAR), help="The company (or market index) to analyse. "
                              "Choose 'Other' to type any Yahoo Finance ticker.")
        if POPULAR[choice] is None:
            display_name = None
            ticker = st.text_input("Type a ticker symbol", "RELIANCE.NS", help="Indian stocks need .NS at the end "
                                   "(e.g. TCS.NS). US stocks need just the code (e.g. AAPL).")
        else:
            ticker, display_name = POPULAR[choice], choice
            st.caption(f"Ticker symbol: **{ticker}**")
        left, right = st.columns(2)
        start = left.date_input("Start date", start, min_value=dt.date(1990, 1, 1), max_value=end,
                                help="First day of price history to use. More years give the model more examples to learn from.")
        end = right.date_input("End date", end, min_value=dt.date(1990, 1, 1), max_value=end,
                               help="Last day of price history to use. Usually today.")
        refresh = st.checkbox("Download fresh data", help="Off: reuse the prices saved the first time, so results "
                              "stay exactly the same. On: download the latest prices from Yahoo Finance again.")
        if start >= end:
            st.error("The start date must be before the end date.")
    else:
        upload = st.file_uploader("Upload a CSV file with Date, Open, High, Low, Close and Volume columns", type="csv")
        ticker = display_name = st.text_input("Name to display", "My Stock")
    st.header("Model")
    model_choice = st.selectbox("Model to use", ["Auto (best on validation)"] + MODEL_NAMES[::-1],
                                help="Auto picks the model with the best accuracy on the validation period.")
    chosen = None if model_choice.startswith("Auto") else model_choice
    st.header("Backtest settings")
    threshold = st.slider("Buy when P(up) is above", 0.50, 0.70, 0.50, 0.01, help="In the backtest, the stock is "
                          "held for the next day only when the model's probability of a rise is above this value. "
                          "Higher means fewer, more confident trades.")
    cost_pct = st.number_input("Transaction cost per trade (%)", 0.0, 1.0, 0.10, 0.05, help="Fee charged in the "
                               "backtest each time the stock is bought or sold (brokerage and taxes). "
                               "0.10% means ₹1 for every ₹1,000 traded.")
    run = st.button("Run analysis", type="primary")

cache = st.cache_data(show_spinner=False)
cached_company = st.cache_data(show_spinner=False, ttl=24 * 3600)(get_company_info)
cached_news = st.cache_data(show_spinner=False, ttl=3600)(get_news_sentiment)
step_models = cache(run_pipeline)
step_walk_forward = cache(walk_forward)


@cache
def get_yahoo(t, s, e, refresh=False):
    return get_prices(t, str(s), str(e), refresh=refresh)


@cache
def step_features(df):
    return model_table(build_features(df))


@cache
def step_predict(data, latest, model_name):
    p_up, fitted = predict_next_day(data, latest, model_name)
    return p_up, explain_prediction(fitted, data, latest)


def full_analysis(df, chosen, status=None):
    def say(msg):
        if status:
            status.update(label=msg)
            status.write(msg)
    say("Step 2 of 4: Engineering 15 features from price history…")
    data, latest = step_features(df)
    say("Step 3 of 4: Training and testing 4 machine-learning models…")
    res = step_models(data, chosen)
    say("Step 4 of 4: Predicting the next session and explaining the prediction…")
    return (data, latest, res, *step_predict(data, latest, res["selected_model"]))

if not run and "df" not in st.session_state:
    st.info("Choose a stock in the sidebar and click **Run analysis**. "
            "On a phone, tap **»** at the top left to open the sidebar.")
    st.stop()

status, status_slot = None, st.container()
if run:
    status = status_slot.status("Step 1 of 4: Loading price data…", expanded=False)
    status.write("Step 1 of 4: Loading price data…")
    try:
        if source == "Yahoo Finance" and refresh:
            get_yahoo.clear()
        df = get_yahoo(ticker, start, end, refresh) if source == "Yahoo Finance" else load_csv(upload) if upload else None
    except Exception as e:
        status.update(label="Could not load the data", state="error")
        st.error(f"Could not load the data. {e}")
        st.stop()
    if df is None:
        status.update(label="No file uploaded", state="error")
        st.warning("Please upload a CSV file.")
        st.stop()
    st.session_state.update(df=df, name=display_name or ticker.upper(),
                            ticker=ticker.strip().upper() if source == "Yahoo Finance" else None)

df, name, yahoo_ticker = st.session_state.df, st.session_state.name, st.session_state.get("ticker")
try:
    data, latest, res, p_up, expl = full_analysis(df, chosen, status)
except ValueError as e:
    if status:
        status.update(label="Not enough data", state="error")
    st.error(str(e))
    st.stop()
if status:
    status.update(label="Pipeline complete — 4 models trained and tested", state="complete")

info, best, last_date = data_summary(df), res["selected_model"], latest.index[-1]
acc_best, acc_base = res["test_table"].loc[best, "accuracy"], res["test_table"].loc[BASELINE, "accuracy"]
is_baseline, up = best == BASELINE, p_up >= 0.5
is_index = (yahoo_ticker or "").startswith("^")

if yahoo_ticker and not is_baseline:
    try:
        log_prediction(name, yahoo_ticker, last_date, best, p_up)
    except Exception:
        pass

model_sub = ("Chosen automatically" if chosen is None else
             f"Auto would pick: {res['auto_choice']}" if chosen != res["auto_choice"] else "Chosen by you")
for col, html in zip(st.columns(4), [
    card("Stock", name, "The company or index being analysed."),
    card("Trading days", f"{info['rows']:,}", "Number of days the market was open in the chosen date range. "
         "All of them are used to train and test the models.", f"{info['start']} → {info['end']}"),
    card("Selected model", best, "The machine-learning model used for the results and the prediction.", model_sub),
    card("Test accuracy", pct(acc_best), "How often the model predicted the direction correctly on the test period "
         "(dates it never saw). The baseline always guesses the most common direction; a useful model should beat it.",
         f"Baseline: {pct(acc_base)}"),
]):
    col.markdown(html, unsafe_allow_html=True)
st.write("")

t_over, t_price, t_model, t_pred, t_news, t_comp, t_about = st.tabs(
    ["Overview", "Prices & indicators", "Model results", "Prediction & explanation",
     "News sentiment", "Compare stocks", "About the project"])

with t_over:
    md(how_to("a quick summary of the stock, the model's prediction and how reliable the model has been. "
              "The other tabs show the details."))
    company = cached_company(yahoo_ticker) if yahoo_ticker else {}
    currency = "" if is_index else company.get("currency", "INR" if (yahoo_ticker or "").endswith((".NS", ".BO")) else "")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Company", help="Basic facts from Yahoo Finance.")
        if company:
            st.markdown(f"**{company.get('longName', company.get('shortName', name))}**")
            low, high = company.get("fiftyTwoWeekLow"), company.get("fiftyTwoWeekHigh")
            facts = {"Sector": company.get("sector", "—"), "Industry": company.get("industry", "—"),
                     "Country": company.get("country", "—"), "Market value": money(company.get("marketCap"), currency),
                     "52-week range": f"{price(low, currency)} – {price(high, currency)}" if low and high else "—",
                     "P/E ratio": f"{company['trailingPE']:.1f}" if company.get("trailingPE") else "—"}
            st.table(pd.DataFrame(facts.items(), columns=["", "Value"]).set_index(""))
            if company.get("longBusinessSummary"):
                with st.expander("About the company"):
                    st.write(company["longBusinessSummary"])
        else:
            st.caption("Yahoo Finance has no company details for this ticker." if yahoo_ticker
                       else "Company details are available for Yahoo Finance stocks only.")
        st.metric("Last close" + (" (points)" if is_index else ""), price(df["Close"].iloc[-1], currency),
                  f"{df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1:+.2%} vs previous day",
                  help=f"Closing price on {last_date.date()}, the last day in the data.")
    with right:
        st.subheader("Model's view", help="The machine-learning prediction for the next trading session.")
        md(prediction_box(up, GREY if is_baseline else GREEN if up else RED, f"Next session after {last_date.date()}",
                          "" if is_baseline else f"Probability of a higher close: <b>{pct(p_up)}</b>",
                          f"<span style='opacity:0.85;font-size:0.85rem'>Model: {best}</span>"))
        st.write("")
        st.markdown(f"On unseen test dates this model was right **{pct(acc_best)}** of the time and "
                    f"**{'beat' if acc_best > acc_base else 'did not beat'}** the simple baseline ({pct(acc_base)}). "
                    "Close to 50% means the prediction is not much better than a coin flip.")

with t_price:
    md(how_to("the top chart shows daily prices with 20- and 50-day averages; the two smaller charts show momentum "
              "indicators. Hover over **(?)** next to a chart title for an explanation, and drag on a chart to zoom in."))
    days = st.slider("Days to show", 60, min(len(df), 2500), min(500, len(df)),
                     help="Only changes how much of the chart is shown. It does not affect the model.")
    ind = add_indicators(df).iloc[-days:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=list(CHART_HELP))
    fig.add_trace(go.Candlestick(x=ind.index, open=ind["Open"], high=ind["High"], low=ind["Low"],
                                 close=ind["Close"], name="Price"), 1, 1)
    for col, label, row, colour in [("sma20", "SMA 20", 1, None), ("sma50", "SMA 50", 1, None),
                                    ("rsi14", "RSI", 2, "purple"), ("macd_line", "MACD", 3, None),
                                    ("macd_signal_line", "Signal", 3, None)]:
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col], name=label, line=dict(width=1, color=colour)), row, 1)
    for level in (70, 30):
        fig.add_hline(y=level, line_dash="dash", line_color="grey", row=2, col=1)
    fig.update_layout(height=750, xaxis_rangeslider_visible=False, margin=dict(t=40, b=20))
    for ann in fig.layout.annotations:
        ann.hovertext, ann.text = CHART_HELP[ann.text], f"{ann.text}  <b>(?)</b>"
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("What do these charts show?"):
        st.markdown("\n".join(f"- **{k}:** {v.replace('<br>', ' ')}" for k, v in CHART_HELP.items()))
    st.download_button("Download price and indicator data (CSV)", add_indicators(df).round(4).to_csv().encode(),
                       file_name=f"{name}_prices_indicators.csv", mime="text/csv")

with t_model:
    md(how_to("four models were trained on older data and scored on newer data they had never seen. Compare each "
              "model with the **baseline**, which always guesses the most common direction. The **test** scores "
              "are the honest final result."))
    st.subheader("Chronological split", help="The data is split by date, never randomly, so the model is always "
                 "tested on days after the ones it learned from.")
    st.table(pd.DataFrame([(k.capitalize(), s.date(), e.date(), n) for k, (s, e, n) in res["periods"].items()],
                          columns=["Period", "From", "To", "Days"]).set_index("Period"))
    left, right = st.columns(2)
    left.subheader("Validation (used to choose the model)",
                   help="Scores on the middle period. Auto mode picks the model with the highest accuracy here.")
    left.dataframe(nice_table(res["val_table"]))
    right.subheader("Test (unseen later dates)",
                    help="Scores on the newest period, used once at the end. This is the honest final result.")
    right.dataframe(nice_table(res["test_table"]))

    st.plotly_chart(bar_chart(["Baseline", "Logistic<br>Regression", "Random<br>Forest", "Gradient<br>Boosting"],
                              {"Validation": res["val_table"].loc[MODEL_NAMES, "accuracy"],
                               "Test": res["test_table"].loc[MODEL_NAMES, "accuracy"]},
                              "Accuracy of each model", y_max=0.7), use_container_width=True)
    st.caption("Validation scores are used to choose the model; test scores are the honest final result. "
               "The dashed line marks 50% — the same as guessing with a coin flip.")
    if not res["beats_baseline_on_validation"]:
        st.warning("No ML model beat the simple baseline on the validation period. "
                   "This is a common and honest result when predicting daily stock direction.")

    left, right = st.columns(2)
    cm = res["confusion_matrix"]
    fig = go.Figure(go.Heatmap(z=cm, x=["Predicted Down", "Predicted Up"], y=["Actual Down", "Actual Up"],
                               text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title=f"Confusion matrix — {best}", height=380, yaxis=dict(autorange="reversed"))
    left.plotly_chart(fig, use_container_width=True)
    fi = res["feature_importance"].iloc[::-1]
    fig = go.Figure(go.Bar(x=fi.values, y=[FEATURE_NAMES[f] for f in fi.index], orientation="h"))
    fig.update_layout(title="Permutation feature importance", height=380, xaxis_title="Drop in accuracy when shuffled")
    right.plotly_chart(fig, use_container_width=True)
    st.caption("**Accuracy:** share of days predicted correctly. **Precision:** when the model said 'up', how often it "
               "was right. **Recall:** of all the real 'up' days, how many the model caught. **F1 score:** a balance "
               "of precision and recall.")
    st.caption("**Confusion matrix:** counts of right and wrong predictions on the test period. The top-left and "
               "bottom-right boxes are correct predictions. **Feature importance:** how much accuracy drops when one "
               "clue is shuffled — longer bars mattered more. It explains how the model behaves; it does not prove causation.")

    st.subheader("Walk-forward test (stricter evaluation)", help="Instead of one test period, the model is re-trained "
                 "and tested on 5 consecutive time blocks. Each time it learns only from earlier data. Steady results "
                 "across blocks are more trustworthy.")
    wf_model = "Gradient Boosting" if is_baseline else best
    st.caption(f"Re-trains **{wf_model}** five times, each time on all earlier data, and tests it on the next block of time.")
    wf_key = (name, info["start"], info["end"], wf_model)
    if st.button("Run walk-forward test"):
        with st.spinner("Training 5 times…"):
            st.session_state.wf = (wf_key, step_walk_forward(data, wf_model))
    if st.session_state.get("wf", (None,))[0] == wf_key:
        wf = st.session_state.wf[1]
        acc, base = wf["Model accuracy"], wf["Baseline accuracy"]
        a, b, c = st.columns(3)
        a.metric("Average accuracy", pct(acc.mean()), help="Average over the 5 test blocks.")
        b.metric("Variation (±)", pct(acc.std()), help="Standard deviation: how much accuracy changes from block to block.")
        c.metric("Blocks that beat the baseline", f"{int((acc > base).sum())} of {len(wf)}")
        folds = [f"Fold {i}<br>{r['Test from']:%b %Y}–<br>{r['Test to']:%b %Y}" for i, r in wf.iterrows()]
        st.plotly_chart(bar_chart(folds, {wf_model: acc, "Baseline": base}, "Accuracy in each walk-forward block",
                                  height=400, y_max=0.75), use_container_width=True)
        st.dataframe(wf.assign(**{c: wf[c].map(pct) for c in ("Model accuracy", "Baseline accuracy")}),
                     use_container_width=True)

    st.subheader("Historical backtest (test period)", help="A pretend trading game on past data: hold the stock when "
                 "the model is confident it will rise, otherwise stay in cash. Compared with simply buying and holding.")
    bt = backtest(res["test_df"], res["test_prob"], threshold, cost_pct / 100)
    fig = go.Figure([go.Scatter(x=bt.index, y=bt[c], name=c) for c in bt.columns])
    fig.update_layout(height=380, yaxis_title="Growth of ₹1 / $1")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A historical simulation including transaction costs. Past results do not promise future profit.")

    st.subheader("Downloads")
    st.caption("To save any chart as an image, hover over it and click the camera icon in its top-right corner.")
    summary = "\n".join([
        f"StockSense results for {name}", f"Data: {info['start']} to {info['end']} ({info['rows']} trading days)",
        "", "Chronological split:",
        *[f"  {k.capitalize():<11} {s.date()} to {e.date()} ({n} days)" for k, (s, e, n) in res["periods"].items()],
        "", "Validation metrics:", nice_table(res["val_table"]).to_string(),
        "", "Test metrics:", nice_table(res["test_table"]).to_string(), "",
        f"Selected model: {best} ({'chosen automatically' if chosen is None else 'chosen by user'})",
        "Confusion matrix (rows = actual Down/Up, columns = predicted Down/Up):", str(cm), "",
        f"Backtest (buy when P(up) > {threshold:.2f}, cost {cost_pct:.2f}% per trade): model strategy "
        f"{bt['Model strategy'].iloc[-1]:.3f}, buy and hold {bt['Buy and hold'].iloc[-1]:.3f}",
        f"Next session after {last_date.date()}: P(up) = {pct(p_up)}", "",
        "Educational project only - not financial advice."])
    scores = pd.concat({"Validation": nice_table(res["val_table"]), "Test": nice_table(res["test_table"])}, names=["Period"])
    for col, (label, content, file, mime) in zip(st.columns(3), [
        ("Model scores (CSV)", scores.to_csv(), "model_scores.csv", "text/csv"),
        ("Backtest results (CSV)", bt.round(4).to_csv(), "backtest.csv", "text/csv"),
        ("Full summary (TXT)", summary, "summary.txt", "text/plain")]):
        col.download_button(label, content.encode(), file_name=f"{name}_{file}", mime=mime)

with t_pred:
    md(how_to("the coloured box is the model's prediction for the next trading day. The chart below explains "
              "**why**: which clues pushed the prediction towards higher (green) or lower (red)."))
    st.subheader(f"Next session after {last_date.date()}", help="The next trading day after the last day in the data. "
                 "The prediction uses that last day's clues.")
    if is_baseline:
        md(prediction_box(up, GREY, f"Model: {best}"))
        st.write("")
        st.warning("The baseline always predicts the most common direction from past data, so it does not give "
                   "a real probability or an explanation. Choose an ML model in the sidebar to see one.")
    else:
        md(prediction_box(up, GREEN if up else RED, f"Probability of a higher close: <b>{pct(p_up)}</b> · Model: {best}"))
        st.write("")
        st.progress(p_up, text=f"Estimated probability of a higher close: {pct(p_up)}")
        st.info(f"Probabilities close to 50% mean the model is unsure. On the test period this model was right "
                f"{pct(acc_best)} of the time.")

        st.subheader("Why did the model predict this?", help="For each clue, today's value is swapped for its typical "
                     "(median) value and the prediction is re-calculated. The change shows how much that clue pushed "
                     "the prediction up or down.")
        top = expl.head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=top.values * 100, y=[FEATURE_NAMES[f] for f in top.index], orientation="h",
                               marker_color=[GREEN if v > 0 else RED for v in top.values],
                               text=[f"{v * 100:+.1f}" for v in top.values], textposition="outside"))
        fig.update_layout(height=380, title="Top 8 clues behind this prediction",
                          xaxis_title="Effect on probability of a higher close (percentage points)")
        st.plotly_chart(fig, use_container_width=True)
        lines = [f"- **{FEATURE_NAMES[f]}** is {fmt_feature(f, latest[f].iloc[0])} (typical: {fmt_feature(f, data[f].median())}) "
                 f"→ pushed the prediction **towards {'higher' if v > 0 else 'lower'}** by {abs(v) * 100:.1f} points."
                 for f, v in expl.head(3).items() if abs(v) >= 0.001]
        if lines:
            st.markdown("**In plain words:**\n" + "\n".join(lines))
        st.caption("This explains the model's reasoning for one day. It does not mean these clues truly cause price moves.")

    st.subheader("Prediction history", help="Every prediction is saved. Once the real result is known (after the next "
                 "trading day, with 'Download fresh data' ticked), it is checked here.")
    if not yahoo_ticker:
        st.caption("Prediction history is kept for Yahoo Finance stocks only.")
    else:
        hist = prediction_history(yahoo_ticker, df)
        known = hist[hist["Correct?"].isin(["Yes", "No"])]
        if len(known):
            st.markdown(f"Correct so far: **{(known['Correct?'] == 'Yes').sum()} of {len(known)}** checked predictions.")
        if len(hist):
            st.dataframe(hist.drop(columns=["Ticker"]).assign(**{"P(up)": hist["P(up)"].map(pct)}),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("No saved predictions for this stock yet.")
        st.caption("To check a prediction, come back after the next trading day, tick **Download fresh data** and run the analysis again.")

with t_news:
    md(how_to("recent headlines about this company, each scored as **positive**, **negative** or **neutral** using a "
              "finance word list (a simple NLP method). This is extra context only — the price models do not use it, "
              "because Yahoo does not provide years of old news to train on."))
    items = None
    if not yahoo_ticker:
        st.caption("News is available for Yahoo Finance stocks only.")
    else:
        try:
            items = cached_news(yahoo_ticker)
        except Exception as err:
            st.warning(f"Could not load news right now. {err}")
        if items == []:
            st.caption("Yahoo Finance has no recent headlines for this ticker.")
    if items:
        avg = float(np.mean([n["score"] for n in items]))
        count = lambda s: sum(n["sentiment"] == s for n in items)
        a, b, c, d = st.columns(4)
        a.metric("Overall news mood", "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral",
                 f"Average score {avg:+.2f}", delta_color="off",
                 help="Average of all headline scores, from -1 (very negative) to +1 (very positive).")
        b.metric("Positive headlines", count("Positive"))
        c.metric("Negative headlines", count("Negative"))
        d.metric("Neutral headlines", count("Neutral"))
        colours = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
        lines = []
        for n in items:
            title = n["title"].replace("[", "(").replace("]", ")")
            link = f"[{title}]({n['link']})" if n["link"] else title
            when = n["time"].strftime("%d %b %Y") if n["time"] else ""
            lines.append(f"- :{colours[n['sentiment']]}[**{n['sentiment']}** ({n['score']:+.1f})] {link}  \n"
                         f"  <span style='opacity:0.7;font-size:0.85rem'>{n['publisher']} · {when}</span>")
        md("\n".join(lines))
        st.caption("Word-list scoring can misread sarcasm or complex sentences. A larger project could use a "
                   "trained language model (for example FinBERT) instead.")

with t_comp:
    md(how_to("runs the whole pipeline on several stocks so you can see whether the approach works better on some "
              "stocks than others. Bars above the dashed 50% line are better than a coin flip."))
    st.caption("Uses the date range and backtest settings from the sidebar. Each new stock takes a few seconds the first time.")
    options = [k for k, v in POPULAR.items() if v]
    picked = st.multiselect("Stocks to compare", options, max_selections=10, help="Pick up to 10 stocks.", default=[
        "Reliance Industries (India)", "Tata Consultancy Services - TCS (India)", "HDFC Bank (India)",
        "Infosys (India)", "Apple (USA)"])
    if st.button("Compare", type="primary", disabled=not picked):
        rows, progress = [], st.progress(0.0, text="Starting…")
        for i, label in enumerate(picked):
            progress.progress(i / len(picked), text=f"Analysing {label} ({i + 1} of {len(picked)})…")
            try:
                r = full_analysis(get_yahoo(POPULAR[label], start, end), None)[2]
                acc, base = r["test_table"].loc[r["selected_model"], "accuracy"], r["test_table"].loc[BASELINE, "accuracy"]
                b = backtest(r["test_df"], r["test_prob"], threshold, cost_pct / 100).iloc[-1]
                rows.append({"Stock": label, "Model (auto)": r["selected_model"], "Test accuracy": acc,
                             "Baseline": base, "Difference": acc - base, "Beats baseline": "Yes" if acc > base else "No",
                             "Model strategy (₹1 →)": round(b["Model strategy"], 3),
                             "Buy and hold (₹1 →)": round(b["Buy and hold"], 3)})
            except Exception as err:
                rows.append({"Stock": label, "Model (auto)": f"Error: {err}"})
        progress.empty()
        st.session_state.compare = pd.DataFrame(rows).set_index("Stock")

    if "compare" in st.session_state:
        comp = st.session_state.compare
        st.dataframe(comp.assign(**{c: comp[c].map(lambda v, c=c: pct(v, signed=c == "Difference"))
                                    for c in ("Test accuracy", "Baseline", "Difference") if c in comp}),
                     use_container_width=True)
        ok = comp.dropna(subset=["Test accuracy"]) if "Test accuracy" in comp else comp.iloc[0:0]
        if len(ok):
            st.markdown(f"The model beat the baseline on **{(ok['Beats baseline'] == 'Yes').sum()} of {len(ok)}** stocks. "
                        f"Average test accuracy: **{pct(ok['Test accuracy'].mean())}** "
                        f"(baseline average: {pct(ok['Baseline'].mean())}).")
            st.plotly_chart(bar_chart(["<br>".join(textwrap.wrap(n, 16)) for n in ok.index],
                                      {"Model (test accuracy)": ok["Test accuracy"], "Baseline": ok["Baseline"]},
                                      "Test accuracy by stock", height=460, labels=False, margin=dict(b=90)),
                            use_container_width=True)
            st.caption("The dashed line marks 50% — the same as guessing with a coin flip.")
        st.download_button("Download comparison (CSV)", comp.round(3).to_csv().encode(),
                           file_name="stock_comparison.csv", mime="text/csv")

with t_about:
    md(how_to("how the project works from start to finish — useful for the report and the viva."))
    table = lambda rows, cols: st.table(pd.DataFrame(rows, columns=cols).set_index(cols[0]))
    st.subheader("Machine-learning pipeline")
    md(ui.flow([("1. Market data", "Yahoo Finance or CSV"), ("2. Pre-processing", "clean, sort, round"),
                ("3. Feature engineering", "15 indicators"), ("4. Model training", "4 models"),
                ("5. Evaluation", "chronological + walk-forward"), ("6. Prediction", "with explanation"),
                ("7. Dashboard", "this website")]))
    st.subheader("Problem definition")
    st.markdown("**Type:** supervised binary classification. **Target:** 1 if the next session's close is higher than "
                "today's close, otherwise 0. **Input:** 15 features calculated only from past prices (no look-ahead).")
    st.subheader("Features")
    table([("return_1d / 5d / 10d", "Percentage price change over the last 1, 5 and 10 days (momentum)."),
           ("close_vs_sma10 / 20 / 50", "How far the price is above or below its 10-, 20- and 50-day average."),
           ("sma10_vs_sma50", "Short-term average compared with long-term average (trend direction)."),
           ("rsi14", "Relative Strength Index over 14 days (0–100 momentum score)."),
           ("macd / macd_signal / macd_hist", "Moving Average Convergence Divergence (trend speed), scaled by price."),
           ("volatility_10d / 20d", "How jumpy daily returns have been (standard deviation)."),
           ("volume_change", "Change in trading volume compared with the previous day."),
           ("high_low_range", "Size of the day's price range compared with the closing price.")], ["Feature", "Meaning"])
    st.subheader("Model card")
    table([(BASELINE, "Always predicts the most common direction in the training data", "—"),
           ("Logistic Regression", "Linear model on standardised features", "C = 0.5"),
           ("Random Forest", "300 decision trees that vote", "max depth 5, min 20 samples per leaf"),
           ("Gradient Boosting", "Trees built one after another, each fixing earlier errors",
            "200 rounds, learning rate 0.05, max depth 3")], ["Model", "How it works", "Main settings"])
    st.subheader("Evaluation methods")
    st.markdown("""
- **Chronological split:** 70% train → 15% validation (model selection) → 15% test (final result, used once).
- **Walk-forward testing:** 5 expanding time windows, each trained only on earlier data.
- **Metrics:** accuracy, precision, recall, F1 score and confusion matrix, always compared with the baseline.
- **Explainability:** permutation feature importance (whole model) and what-if explanations (single prediction).
- **Backtest:** historical simulation with transaction costs, compared with buy-and-hold.
- **Reproducibility:** downloaded prices are saved and re-used, and models use a fixed random seed.""")
    st.subheader("Technology stack")
    table([("Python", "Core implementation"), ("Pandas, NumPy", "Data cleaning and feature engineering"),
           ("scikit-learn", "Machine-learning models and evaluation"), ("yfinance", "Market data, company info and news"),
           ("Plotly", "Interactive charts"), ("Streamlit", "Web dashboard")], ["Tool", "Used for"])
    st.subheader("Limitations")
    st.markdown("""
- Markets are noisy and react to news and events that are not in price data.
- Results vary across stocks, time periods and market conditions.
- Directional accuracy does not mean a strategy is profitable.
- News sentiment uses recent headlines only and a simple word list, so it is not part of the trained models.
- Data availability and permitted use depend on the data provider (Yahoo Finance terms).""")
