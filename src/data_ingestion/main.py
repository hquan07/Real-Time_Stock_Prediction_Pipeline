import time
import logging
from typing import Dict, Any

from .fetcher.tickers import get_ticker_list
from .fetcher.price import fetch_price
from .fetcher.dividends import fetch_dividends
from .fetcher.earnings import fetch_earnings
from .fetcher.splits import fetch_splits
from .fetcher.financials import fetch_financials

from .kafka_producer.send_data import send_to_kafka
from .config.fetcher_config import FETCH_INTERVAL, MAX_RETRIES, RETRY_WAIT
from .config.kafka_config import KAFKA_TOPIC

from .schema.schema_validator import validate_stock_message


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("data_ingestion")


# -------------------------------------------------------------------
# Build message payload
# -------------------------------------------------------------------
def build_stock_message(ticker: str) -> Dict[str, Any]:
    """
    Fetch all stock-related data for a ticker
    and build a Kafka-ready payload matching Avro schema.
    """
    try:
        price = fetch_price(ticker)
        dividends = fetch_dividends(ticker)
        earnings = fetch_earnings(ticker)
        splits = fetch_splits(ticker)
        financials = fetch_financials(ticker)

        message = {
            "ticker": ticker,
            "timestamp": int(time.time() * 1000),

            "price": {
                "open": float(price.get("open", 0)),
                "high": float(price.get("high", 0)),
                "low": float(price.get("low", 0)),
                "close": float(price.get("close", 0)),
                "volume": int(price.get("volume", 0))
            } if price else None,

            "dividends": float(dividends) if dividends is not None else None,
            "splits": float(splits) if splits is not None else None,

            "earnings": {
                "eps_actual": float(earnings.get("actual", 0)),
                "eps_estimate": float(earnings.get("estimate", 0)),
                "quarter": earnings.get("quarter")
            } if earnings else None,

            "financials": {
                "total_revenue": int(financials.get("total_revenue", 0)),
                "net_income": int(financials.get("net_income", 0)),
                "operating_income": int(financials.get("operating_income", 0))
            } if financials else None
        }

        return message

    except Exception as e:
        logger.exception(f"❌ Failed to build message for ticker={ticker}")
        raise e


# -------------------------------------------------------------------
# Main pipeline runner
# -------------------------------------------------------------------
def run_pipeline() -> None:
    """
    Run YFinance → Kafka ingestion pipeline.
    """
    tickers = get_ticker_list()
    logger.info(f"🚀 Starting ingestion pipeline for {len(tickers)} tickers")

    for ticker in tickers:
        attempt = 0

        while attempt < MAX_RETRIES:
            try:
                logger.info(f"➡ Fetching data for {ticker}")
                message = build_stock_message(ticker)

                # 🔍 STEP 2.1.2 – Avro schema validation
                validate_stock_message(message)

                # 🚀 Send to Kafka
                send_to_kafka(
                    topic=KAFKA_TOPIC,
                    data=message
                )

                logger.info(f"✔ Sent data for {ticker}")
                break

            except Exception as e:
                attempt += 1
                logger.warning(
                    f"Retry {attempt}/{MAX_RETRIES} for {ticker} due to error: {e}"
                )
                time.sleep(RETRY_WAIT)

        time.sleep(0.5)  # Avoid API rate limits

    logger.info("✅ Data ingestion pipeline completed")


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        logger.warning("🛑 Pipeline interrupted by user")
    except Exception as e:
        logger.exception("🔥 Fatal error in ingestion pipeline")