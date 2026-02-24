"""
Airflow script to run data fetching pipeline.
Called by yfinance_pipeline_dag.py
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from loguru import logger


def run_fetch(
    tickers: list = None,
    limit: int = 10,
    save_to_db: bool = True,
    send_to_kafka: bool = True,
) -> dict:
    """
    Run the data fetching pipeline.
    
    Args:
        tickers: List of tickers to fetch (None = use default list)
        limit: Maximum number of tickers to process
        save_to_db: Whether to save data to PostgreSQL
        send_to_kafka: Whether to send data to Kafka
    
    Returns:
        Dict with execution stats.
    """
    from src.data_ingestion.fetcher.tickers import get_ticker_list
    from src.data_ingestion.fetcher.price import fetch_price
    from src.data_ingestion.fetcher.dividends import fetch_dividends
    from src.database.write_raw import write_price_history, write_dividend

    start_time = datetime.now()
    stats = {
        "tickers_processed": 0,
        "success": 0,
        "failed": 0,
        "start_time": start_time.isoformat(),
    }

    # Get tickers
    if tickers is None:
        try:
            tickers = get_ticker_list()[:limit]
        except Exception as e:
            logger.warning(f"Could not fetch ticker list: {e}")
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"][:limit]

    logger.info(f"🚀 Starting data fetch for {len(tickers)} tickers")

    for ticker in tickers:
        try:
            logger.info(f"Fetching {ticker}...")

            # Fetch price
            price = fetch_price(ticker)

            if price:
                stats["success"] += 1

                if save_to_db:
                    write_price_history(
                        ticker=ticker,
                        date_val=datetime.now().date(),
                        open_price=price.get("open"),
                        high=price.get("high"),
                        low=price.get("low"),
                        close=price.get("close"),
                        volume=price.get("volume"),
                    )

                # Fetch and save dividends
                dividend = fetch_dividends(ticker)
                if dividend and save_to_db:
                    write_dividend(ticker, datetime.now().date(), dividend)

            else:
                stats["failed"] += 1
                logger.warning(f"No data for {ticker}")

        except Exception as e:
            stats["failed"] += 1
            logger.error(f"Error fetching {ticker}: {e}")

        stats["tickers_processed"] += 1

    end_time = datetime.now()
    stats["end_time"] = end_time.isoformat()
    stats["duration_seconds"] = (end_time - start_time).total_seconds()

    logger.info(
        f"✅ Fetch complete: {stats['success']}/{stats['tickers_processed']} successful "
        f"in {stats['duration_seconds']:.1f}s"
    )

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data fetch pipeline")
    parser.add_argument("--limit", type=int, default=10, help="Max tickers to fetch")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to fetch")
    parser.add_argument("--no-db", action="store_true", help="Skip database save")
    parser.add_argument("--no-kafka", action="store_true", help="Skip Kafka send")

    args = parser.parse_args()

    result = run_fetch(
        tickers=args.tickers,
        limit=args.limit,
        save_to_db=not args.no_db,
        send_to_kafka=not args.no_kafka,
    )

    print(f"Result: {result}")
