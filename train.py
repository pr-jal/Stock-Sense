import argparse
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import data_summary, get_prices, load_csv
from features import add_indicators, build_features, model_table
from model import backtest, predict_next_day, run_pipeline


def main():
    p = argparse.ArgumentParser(description="StockSense: next-day up/down prediction")
    p.add_argument("--ticker", default="RELIANCE.NS", help="Yahoo Finance symbol")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=dt.date.today().isoformat())
    p.add_argument("--csv", help="Use a local CSV instead of downloading")
    p.add_argument("--name", help="Display name when using --csv")
    p.add_argument("--refresh", action="store_true", help="Download fresh data instead of reusing the saved copy")
    args = p.parse_args()

    if args.csv:
        df, name, source = load_csv(args.csv), args.name or Path(args.csv).stem, f"CSV file: {args.csv}"
    else:
        df = get_prices(args.ticker, args.start, args.end, refresh=args.refresh)
        name, source = args.ticker.upper(), "Yahoo Finance via yfinance (adjusted prices)"
    info = data_summary(df)
    print(f"Loaded {info['rows']} days for {name} ({info['start']} to {info['end']})")

    data, latest = model_table(build_features(df))
    res = run_pipeline(data)
    best = res["selected_model"]
    bt = backtest(res["test_df"], res["test_prob"])
    p_up, _ = predict_next_day(data, latest, best)

    out = Path("results") / name.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    res["val_table"].round(4).to_csv(out / "validation_metrics.csv")
    res["test_table"].round(4).to_csv(out / "test_metrics.csv")

    text = "\n".join([
        f"StockSense results for {name}", f"Data source: {source}",
        f"Date range: {info['start']} to {info['end']}  ({info['rows']} trading days)", "", "Chronological split:",
        *[f"  {k:<11} {s.date()} to {e.date()}  ({n} days)" for k, (s, e, n) in res["periods"].items()],
        "", "Validation metrics (used to choose the model):", res["val_table"].round(4).to_string(), "",
        f"Selected model: {best}", f"Beats baseline on validation: {res['beats_baseline_on_validation']}", "",
        "TEST metrics (unseen later dates, evaluated once):", res["test_table"].round(4).to_string(), "",
        f"Share of 'up' days in test period: {res['up_share_test']:.1%}",
        f"Confusion matrix for {best} (rows = actual Down/Up, cols = predicted Down/Up):",
        str(res["confusion_matrix"]), "",
        f"Historical backtest on test period (0.1% cost per trade): strategy x{bt['Model strategy'].iloc[-1]:.3f}, "
        f"buy-and-hold x{bt['Buy and hold'].iloc[-1]:.3f}", "",
        f"Next session after {latest.index[-1].date()}: P(close higher) = {p_up:.1%}",
        "Educational project only - not financial advice."])
    (out / "summary.txt").write_text(text)
    print("\n" + text)
    save_charts(df, res, bt, best, name, out)
    print(f"\nSaved results and charts to {out}/")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_charts(df, res, bt, best, name, out):
    ind = add_indicators(df).iloc[-500:]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for col, label, lw in [("Close", "Close", 1.2), ("sma20", "SMA 20", 1), ("sma50", "SMA 50", 1)]:
        a1.plot(ind.index, ind[col], label=label, lw=lw)
    a1.set_title(f"{name}: price and moving averages (last 500 days)")
    a1.legend()
    a2.plot(ind.index, ind["rsi14"], color="purple", lw=1)
    for level in (70, 30):
        a2.axhline(level, ls="--", c="grey")
    a2.set_ylabel("RSI 14")
    save(fig, out / "price_indicators.png")

    cm = res["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Down", "Up"])
    ax.set_yticks([0, 1], ["Down", "Up"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix - {best}\n(test period)")
    save(fig, out / "confusion_matrix.png")

    fi = res["feature_importance"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(fi.index, fi.values)
    ax.set(title=f"Permutation feature importance - {best}", xlabel="Drop in test accuracy when shuffled")
    save(fig, out / "feature_importance.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in bt.columns:
        ax.plot(bt.index, bt[col], label=col)
    ax.set(title="Historical backtest on test period (0.1% cost per trade)", ylabel="Growth of 1 unit")
    ax.legend()
    save(fig, out / "backtest.png")

if __name__ == "__main__":
    main()
