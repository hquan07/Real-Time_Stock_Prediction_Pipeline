import os
from datetime import datetime, timedelta

# Sample tickers for testing (can be overridden by environment)
TICKERS = os.getenv("TICKERS", "AAPL,GOOGL,MSFT,AMZN,META").split(",")

FETCH_INTERVAL = os.getenv("FETCH_INTERVAL", "1d")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "5"))

END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAYS)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_WAIT = int(os.getenv("RETRY_WAIT", "2"))