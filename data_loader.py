import pandas as pd

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]


def download_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    ticker = ticker.strip().upper()

    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data was found for '{ticker}'. Check the ticker symbol (Indian stocks "
                         f"need .NS at the end, e.g. RELIANCE.NS) and the dates.")
    df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return clean(df)


def load_csv(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("_", " ")
        if key in ("date", "datetime", "timestamp"):
            rename[col] = "Date"
        elif key in ("adj close", "adjclose"):
            rename[col] = "Adj Close"
        elif key in ("open", "high", "low", "close", "volume"):
            rename[col] = key.capitalize()
    df = df.rename(columns=rename)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")

    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("The data is missing these columns: " + ", ".join(missing))

    df = df[REQUIRED]
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].round(2)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna()
    df = df[(df["Close"] > 0) & (df["Volume"] >= 0)]
    return df


def data_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "start": df.index.min().date().isoformat(),
        "end": df.index.max().date().isoformat(),
    }


def get_prices(ticker: str, start: str, end: str, refresh: bool = False, folder: str = "data") -> pd.DataFrame:
    from pathlib import Path
    ticker = ticker.strip().upper()
    path = Path(folder) / f"{ticker.replace('^', 'INDEX_')}_{start}_{end}.csv"
    if path.exists() and not refresh:
        return load_csv(path)
    df = download_yahoo(ticker, start, end)
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path)
    return df


def get_company_info(ticker: str) -> dict:
    import yfinance as yf
    try:
        info = yf.Ticker(ticker.strip().upper()).info or {}
    except Exception:
        return {}
    keys = ["longName", "shortName", "sector", "industry", "country", "currency", "marketCap",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "trailingPE", "website", "longBusinessSummary"]
    return {k: info.get(k) for k in keys if info.get(k) not in (None, "")}

LOG_COLUMNS = ["Logged at", "Stock", "Ticker", "Prediction for day after", "Model", "P(up)", "Prediction"]


def log_prediction(stock: str, ticker: str, last_date, model: str, p_up: float,
                   path: str = "data/prediction_log.csv") -> None:
    import datetime as dt
    from pathlib import Path
    path = Path(path)
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    key = (ticker, str(last_date.date()), model)
    if ((log["Ticker"] == key[0]) & (log["Prediction for day after"] == key[1]) & (log["Model"] == key[2])).any():
        return
    row = {"Logged at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "Stock": stock, "Ticker": ticker,
           "Prediction for day after": key[1], "Model": model, "P(up)": round(p_up, 4),
           "Prediction": "Higher" if p_up >= 0.5 else "Lower"}
    path.parent.mkdir(exist_ok=True)
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def prediction_history(ticker: str, prices: pd.DataFrame, path: str = "data/prediction_log.csv") -> pd.DataFrame:
    from pathlib import Path
    if not Path(path).exists():
        return pd.DataFrame(columns=LOG_COLUMNS + ["Actual", "Correct?"])
    log = pd.read_csv(path)
    log = log[log["Ticker"] == ticker].copy()
    actual, correct = [], []
    for _, r in log.iterrows():
        day = pd.Timestamp(r["Prediction for day after"])
        later = prices.loc[prices.index > day]
        if day in prices.index and len(later):
            went_up = later["Close"].iloc[0] > prices.loc[day, "Close"]
            actual.append("Higher" if went_up else "Lower")
            correct.append("Yes" if actual[-1] == r["Prediction"] else "No")
        else:
            actual.append("Not known yet")
            correct.append("—")
    log["Actual"], log["Correct?"] = actual, correct
    return log.iloc[::-1]
