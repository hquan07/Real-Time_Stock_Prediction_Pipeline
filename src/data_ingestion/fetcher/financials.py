import yfinance as yf


def fetch_financials(ticker: str):
    """
    Fetch key financial metrics from income statement
    """
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials

        if financials is None or financials.empty:
            return None

        latest_col = financials.columns[0]

        return {
            "total_revenue": (
                int(financials.loc["Total Revenue", latest_col])
                if "Total Revenue" in financials.index
                else None
            ),
            "net_income": (
                int(financials.loc["Net Income", latest_col])
                if "Net Income" in financials.index
                else None
            ),
            "operating_income": (
                int(financials.loc["Operating Income", latest_col])
                if "Operating Income" in financials.index
                else None
            )
        }

    except Exception:
        return None