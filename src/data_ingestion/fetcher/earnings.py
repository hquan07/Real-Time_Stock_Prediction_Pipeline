import yfinance as yf


def fetch_earnings(ticker: str):
    """
    Fetch latest quarterly earnings
    """
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_dates

        if earnings is None or earnings.empty:
            return None

        latest = earnings.iloc[0]

        return {
            "actual": (
                float(latest["Reported EPS"])
                if "Reported EPS" in latest and latest["Reported EPS"] is not None
                else None
            ),
            "estimate": (
                float(latest["EPS Estimate"])
                if "EPS Estimate" in latest and latest["EPS Estimate"] is not None
                else None
            ),
            "quarter": str(latest.name)
        }

    except Exception:
        return None