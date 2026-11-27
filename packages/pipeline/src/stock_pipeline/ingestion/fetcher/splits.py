import yfinance as yf


def fetch_splits(ticker: str):
    """
    Fetch latest stock split ratio
    """
    try:
        stock = yf.Ticker(ticker)
        splits = stock.splits

        if splits.empty:
            return None

        return float(splits.iloc[-1])

    except Exception:
        return None