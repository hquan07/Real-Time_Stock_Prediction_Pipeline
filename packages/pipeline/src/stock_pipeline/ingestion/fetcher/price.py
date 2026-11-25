import yfinance as yf
import pandas as pd


def fetch_price(ticker: str):
    """
    Fetch latest OHLCV price
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")

        if hist.empty:
            return None

        row = hist.iloc[-1]
        event_time = pd.Timestamp(hist.index[-1])
        if event_time.tzinfo is None:
            event_time = event_time.tz_localize("America/New_York")
        event_time = event_time.tz_convert("UTC")

        return {
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
            "event_time_ms": int(event_time.timestamp() * 1000),
        }

    except Exception:
        return None
