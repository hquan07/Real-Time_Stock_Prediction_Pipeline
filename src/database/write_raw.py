"""
Module to write raw stock data to PostgreSQL.
Handles price history, intraday prices, company info, etc.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional

from sqlalchemy import text
from loguru import logger

from .db_connection import get_session


# ---------------------------------------------------------------------------
# Write Company Info
# ---------------------------------------------------------------------------
def write_company(
    ticker: str,
    long_name: str = None,
    sector: str = None,
    industry: str = None,
    country: str = None,
    exchange: str = None,
    website: str = None,
    description: str = None,
) -> bool:
    """
    Insert or update company information.
    Uses UPSERT (ON CONFLICT) to handle duplicates.
    """
    query = text("""
        INSERT INTO companies (
            ticker, long_name, sector, industry, 
            country, exchange, website, description
        )
        VALUES (
            :ticker, :long_name, :sector, :industry,
            :country, :exchange, :website, :description
        )
        ON CONFLICT (ticker) DO UPDATE SET
            long_name = EXCLUDED.long_name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            country = EXCLUDED.country,
            exchange = EXCLUDED.exchange,
            website = EXCLUDED.website,
            description = EXCLUDED.description
    """)

    try:
        with get_session() as session:
            session.execute(query, {
                "ticker": ticker,
                "long_name": long_name,
                "sector": sector,
                "industry": industry,
                "country": country,
                "exchange": exchange,
                "website": website,
                "description": description,
            })
        logger.debug(f"✅ Company {ticker} upserted")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write company {ticker}: {e}")
        return False


# ---------------------------------------------------------------------------
# Write Price History
# ---------------------------------------------------------------------------
def write_price_history(
    ticker: str,
    date_val: date,
    open_price: float = None,
    high: float = None,
    low: float = None,
    close: float = None,
    adj_close: float = None,
    volume: int = None,
) -> bool:
    """
    Insert or update daily price history.
    """
    query = text("""
        INSERT INTO price_history (
            ticker, date, open, high, low, close, adj_close, volume
        )
        VALUES (
            :ticker, :date, :open, :high, :low, :close, :adj_close, :volume
        )
        ON CONFLICT (ticker, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume
    """)

    try:
        with get_session() as session:
            session.execute(query, {
                "ticker": ticker,
                "date": date_val,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": adj_close,
                "volume": volume,
            })
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write price history for {ticker}: {e}")
        return False


def write_price_history_batch(records: List[Dict[str, Any]]) -> int:
    """
    Batch insert price history records.
    
    Args:
        records: List of dicts with keys: ticker, date, open, high, low, close, adj_close, volume
    
    Returns:
        Number of records successfully inserted.
    """
    query = text("""
        INSERT INTO price_history (
            ticker, date, open, high, low, close, adj_close, volume
        )
        VALUES (
            :ticker, :date, :open, :high, :low, :close, :adj_close, :volume
        )
        ON CONFLICT (ticker, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume
    """)

    success_count = 0

    try:
        with get_session() as session:
            for record in records:
                try:
                    session.execute(query, record)
                    success_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Skipping record: {e}")
                    continue

        logger.info(f"✅ Batch inserted {success_count}/{len(records)} price records")
        return success_count

    except Exception as e:
        logger.error(f"❌ Batch price history write failed: {e}")
        return success_count


# ---------------------------------------------------------------------------
# Write Intraday Prices
# ---------------------------------------------------------------------------
def write_intraday_price(
    ticker: str,
    datetime_val: datetime,
    open_price: float = None,
    high: float = None,
    low: float = None,
    close: float = None,
    volume: int = None,
) -> bool:
    """
    Insert or update intraday price data.
    """
    query = text("""
        INSERT INTO intraday_prices (
            ticker, datetime, open, high, low, close, volume
        )
        VALUES (
            :ticker, :datetime, :open, :high, :low, :close, :volume
        )
        ON CONFLICT (ticker, datetime) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
    """)

    try:
        with get_session() as session:
            session.execute(query, {
                "ticker": ticker,
                "datetime": datetime_val,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write intraday price for {ticker}: {e}")
        return False


# ---------------------------------------------------------------------------
# Write Dividends
# ---------------------------------------------------------------------------
def write_dividend(ticker: str, date_val: date, dividend: float) -> bool:
    """Insert or update dividend record."""
    query = text("""
        INSERT INTO dividends (ticker, date, dividend)
        VALUES (:ticker, :date, :dividend)
        ON CONFLICT (ticker, date) DO UPDATE SET
            dividend = EXCLUDED.dividend
    """)

    try:
        with get_session() as session:
            session.execute(query, {
                "ticker": ticker,
                "date": date_val,
                "dividend": dividend,
            })
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write dividend for {ticker}: {e}")
        return False


# ---------------------------------------------------------------------------
# Write Stock Splits
# ---------------------------------------------------------------------------
def write_stock_split(ticker: str, date_val: date, split_ratio: float) -> bool:
    """Insert or update stock split record."""
    query = text("""
        INSERT INTO stock_splits (ticker, date, split_ratio)
        VALUES (:ticker, :date, :split_ratio)
        ON CONFLICT (ticker, date) DO UPDATE SET
            split_ratio = EXCLUDED.split_ratio
    """)

    try:
        with get_session() as session:
            session.execute(query, {
                "ticker": ticker,
                "date": date_val,
                "split_ratio": split_ratio,
            })
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write stock split for {ticker}: {e}")
        return False


# ---------------------------------------------------------------------------
# Write Earnings
# ---------------------------------------------------------------------------
def write_earnings_history(
    ticker: str,
    period: date,
    revenue_actual: float = None,
    earnings_actual: float = None,
    earnings_estimate: float = None,
) -> bool:
    """Insert or update earnings history."""
    query = text("""
        INSERT INTO earnings_history (
            ticker, period, revenue_actual, earnings_actual, earnings_estimate
        )
        VALUES (:ticker, :period, :revenue_actual, :earnings_actual, :earnings_estimate)
        ON CONFLICT (ticker, period) DO UPDATE SET
            revenue_actual = EXCLUDED.revenue_actual,
            earnings_actual = EXCLUDED.earnings_actual,
            earnings_estimate = EXCLUDED.earnings_estimate
    """)

    try:
        with get_session() as session:
            session.execute(query, {
                "ticker": ticker,
                "period": period,
                "revenue_actual": revenue_actual,
                "earnings_actual": earnings_actual,
                "earnings_estimate": earnings_estimate,
            })
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write earnings for {ticker}: {e}")
        return False


# ---------------------------------------------------------------------------
# Write Financial Statements
# ---------------------------------------------------------------------------
def write_income_statement(
    ticker: str,
    period: date,
    total_revenue: float = None,
    gross_profit: float = None,
    operating_income: float = None,
    net_income: float = None,
    ebit: float = None,
    eps: float = None,
    other_items: dict = None,
) -> bool:
    """Insert or update income statement."""
    import json

    query = text("""
        INSERT INTO income_statements (
            ticker, period, total_revenue, gross_profit, 
            operating_income, net_income, ebit, eps, other_financial_items
        )
        VALUES (
            :ticker, :period, :total_revenue, :gross_profit,
            :operating_income, :net_income, :ebit, :eps, :other_items
        )
        ON CONFLICT (ticker, period) DO UPDATE SET
            total_revenue = EXCLUDED.total_revenue,
            gross_profit = EXCLUDED.gross_profit,
            operating_income = EXCLUDED.operating_income,
            net_income = EXCLUDED.net_income,
            ebit = EXCLUDED.ebit,
            eps = EXCLUDED.eps,
            other_financial_items = EXCLUDED.other_financial_items
    """)

    try:
        with get_session() as session:
            session.execute(query, {
                "ticker": ticker,
                "period": period,
                "total_revenue": total_revenue,
                "gross_profit": gross_profit,
                "operating_income": operating_income,
                "net_income": net_income,
                "ebit": ebit,
                "eps": eps,
                "other_items": json.dumps(other_items) if other_items else None,
            })
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write income statement for {ticker}: {e}")
        return False


# ---------------------------------------------------------------------------
# Convenience function for Kafka consumer
# ---------------------------------------------------------------------------
def write_kafka_message(message: Dict[str, Any]) -> bool:
    """
    Write a Kafka stock message to database.
    Handles the message structure from data_ingestion module.
    """
    ticker = message.get("ticker")
    timestamp = message.get("timestamp")

    if not ticker:
        logger.warning("⚠️ Message missing ticker, skipping")
        return False

    # Convert timestamp to datetime
    if timestamp:
        dt = datetime.fromtimestamp(timestamp / 1000)
    else:
        dt = datetime.now()

    # Write price data
    price = message.get("price")
    if price:
        write_intraday_price(
            ticker=ticker,
            datetime_val=dt,
            open_price=price.get("open"),
            high=price.get("high"),
            low=price.get("low"),
            close=price.get("close"),
            volume=price.get("volume"),
        )

    # Write dividends
    dividends = message.get("dividends")
    if dividends:
        write_dividend(ticker, dt.date(), dividends)

    # Write splits
    splits = message.get("splits")
    if splits:
        write_stock_split(ticker, dt.date(), splits)

    logger.debug(f"✅ Kafka message for {ticker} written to DB")
    return True