import numpy as np
import pandas as pd

FEATURES = [
    "return_1d", "return_5d", "return_10d",
    "close_vs_sma10", "close_vs_sma20", "close_vs_sma50", "sma10_vs_sma50",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "volatility_10d", "volatility_20d",
    "volume_change", "high_low_range",
]


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    for n in (10, 20, 50):
        out[f"sma{n}"] = close.rolling(n).mean()
    ema = lambda span: close.ewm(span=span, adjust=False).mean()
    out["macd_line"] = ema(12) - ema(26)
    out["macd_signal_line"] = out["macd_line"].ewm(span=9, adjust=False).mean()
    out["rsi14"] = rsi(close, 14)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_indicators(df)
    close = out["Close"]
    daily_ret = close.pct_change()

    for n in (1, 5, 10):
        out[f"return_{n}d"] = close.pct_change(n)
    for n in (10, 20, 50):
        out[f"close_vs_sma{n}"] = close / out[f"sma{n}"] - 1
    out["sma10_vs_sma50"] = out["sma10"] / out["sma50"] - 1
    out["macd"] = out["macd_line"] / close
    out["macd_signal"] = out["macd_signal_line"] / close
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    for n in (10, 20):
        out[f"volatility_{n}d"] = daily_ret.rolling(n).std()
    out["volume_change"] = np.log1p(out["Volume"]).diff()
    out["high_low_range"] = (out["High"] - out["Low"]) / close

    next_close = close.shift(-1)
    out["target"] = (next_close > close).astype(float).where(next_close.notna())
    return out.replace([np.inf, -np.inf], np.nan)


def model_table(feat: pd.DataFrame):
    usable = feat.dropna(subset=FEATURES)
    labelled = usable.dropna(subset=["target"])
    latest = usable.iloc[[-1]]
    return labelled, latest
