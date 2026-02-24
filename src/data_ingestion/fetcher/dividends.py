import yfinance as yf


def fetch_dividends(ticker: str):
    """
    Fetch latest dividend value
    """
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends

        if dividends.empty:
            return None

        return float(dividends.iloc[-1])

    except Exception:
        return None