"""
Ticker list management for data ingestion.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Default tickers for testing/fallback
DEFAULT_TICKERS = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "WMT"
]


def get_ticker_list():
    """
    Get list of stock tickers to fetch.
    
    Uses TICKERS environment variable if set, otherwise falls back to default list.
    The old NASDAQ URLs are deprecated and no longer work reliably.
    
    Returns:
        List of ticker symbols
    """
    tickers_env = os.getenv("TICKERS", "")
    
    if tickers_env:
        tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]
        logger.info(f"Using {len(tickers)} tickers from environment")
        return tickers
    
    logger.info(f"Using {len(DEFAULT_TICKERS)} default tickers")
    return DEFAULT_TICKERS