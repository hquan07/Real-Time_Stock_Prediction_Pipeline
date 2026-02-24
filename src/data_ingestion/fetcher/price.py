import yfinance as yf


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

        return {
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        }

    except Exception:
        return None