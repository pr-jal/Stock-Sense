import re
from datetime import datetime, timezone

POSITIVE = {
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies",
    "gain", "gains", "rise", "rises", "rising", "record", "high", "highs", "growth", "grow", "grows",
    "profit", "profits", "profitable", "strong", "stronger", "upgrade", "upgrades", "upgraded",
    "outperform", "buy", "bullish", "boost", "boosts", "expands", "expansion", "wins", "win",
    "approval", "approved", "dividend", "partnership", "deal", "launch", "launches", "positive",
    "recovery", "rebound", "rebounds", "optimistic", "top", "tops", "exceeds", "raises", "raised",
}
NEGATIVE = {
    "miss", "misses", "missed", "fall", "falls", "falling", "drop", "drops", "plunge", "plunges",
    "slump", "slumps", "decline", "declines", "loss", "losses", "weak", "weaker", "downgrade",
    "downgrades", "downgraded", "underperform", "sell", "bearish", "cut", "cuts", "lawsuit",
    "probe", "investigation", "fine", "fined", "penalty", "fraud", "scandal", "recall", "layoffs",
    "layoff", "crash", "crashes", "warning", "warns", "risk", "risks", "concern", "concerns",
    "slowdown", "debt", "default", "negative", "pressure", "low", "lows", "tumble", "tumbles",
    "lower", "lowers", "halt", "ban", "delay", "delays",
}
NEGATORS = {"not", "no", "never", "without", "fails", "fail", "failed"}


def score_headline(text: str) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    total = 0
    for i, w in enumerate(words):
        value = 1 if w in POSITIVE else -1 if w in NEGATIVE else 0
        if value and i > 0 and words[i - 1] in NEGATORS:
            value = -value
        total += value
    return max(-1.0, min(1.0, total / 2))


def label(score: float) -> str:
    return "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"


def _parse(item: dict) -> dict:
    c = item.get("content", item)
    title = c.get("title", "")
    publisher = (c.get("provider") or {}).get("displayName") or item.get("publisher", "")
    link = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
            or item.get("link", ""))
    when = c.get("pubDate") or c.get("displayTime")
    if when:
        try:
            when = datetime.fromisoformat(when.replace("Z", "+00:00"))
        except ValueError:
            when = None
    elif item.get("providerPublishTime"):
        when = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc)
    return {"title": title, "publisher": publisher, "link": link, "time": when}


def get_news_sentiment(ticker: str, limit: int = 15) -> list:
    import yfinance as yf
    raw = yf.Ticker(ticker.strip().upper()).news or []
    out = []
    for item in raw[:limit]:
        n = _parse(item)
        if n["title"]:
            n["score"] = score_headline(n["title"])
            n["sentiment"] = label(n["score"])
            out.append(n)
    return out
