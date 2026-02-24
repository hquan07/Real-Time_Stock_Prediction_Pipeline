#!/usr/bin/env python3
"""
Complete Database Populator
Fetches comprehensive stock data from yfinance and populates all database tables
"""

import yfinance as yf
import psycopg2
from datetime import datetime, timedelta
from loguru import logger

# PostgreSQL config  
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "stockdb"
DB_USER = "postgres"
DB_PASSWORD = "Huyquan1607"

# Stocks to fetch
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']


def get_db_connection():
    """Create PostgreSQL connection"""
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )


def populate_company(conn, ticker: str, info: dict):
    """Populate companies table with full company info"""
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO companies (ticker, long_name, sector, industry, country, exchange, website, description)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                long_name = EXCLUDED.long_name,
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                country = EXCLUDED.country,
                exchange = EXCLUDED.exchange,
                website = EXCLUDED.website,
                description = EXCLUDED.description
        """, (
            ticker,
            info.get('longName', ticker),
            info.get('sector'),
            info.get('industry'),
            info.get('country'),
            info.get('exchange'),
            info.get('website'),
            info.get('longBusinessSummary', '')[:500] if info.get('longBusinessSummary') else None
        ))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error populating company {ticker}: {e}")
        conn.rollback()
        return False


def populate_price_history(conn, ticker: str, hist):
    """Populate price_history table"""
    try:
        cur = conn.cursor()
        count = 0
        for date, row in hist.iterrows():
            cur.execute("""
                INSERT INTO price_history (ticker, date, open, high, low, close, adj_close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adj_close = EXCLUDED.adj_close,
                    volume = EXCLUDED.volume
            """, (
                ticker,
                date.date(),
                float(row['Open']) if row['Open'] else None,
                float(row['High']) if row['High'] else None,
                float(row['Low']) if row['Low'] else None,
                float(row['Close']) if row['Close'] else None,
                float(row.get('Adj Close', row['Close'])) if row.get('Adj Close', row['Close']) else None,
                int(row['Volume']) if row['Volume'] else None
            ))
            count += 1
        conn.commit()
        return count
    except Exception as e:
        logger.error(f"Error populating price_history for {ticker}: {e}")
        conn.rollback()
        return 0


def populate_dividends(conn, ticker: str, dividends):
    """Populate dividends table"""
    try:
        cur = conn.cursor()
        count = 0
        for date, amount in dividends.items():
            if amount > 0:
                cur.execute("""
                    INSERT INTO dividends (ticker, date, dividend)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (ticker, date) DO UPDATE SET dividend = EXCLUDED.dividend
                """, (ticker, date.date(), float(amount)))
                count += 1
        conn.commit()
        return count
    except Exception as e:
        logger.error(f"Error populating dividends for {ticker}: {e}")
        conn.rollback()
        return 0


def populate_stock_splits(conn, ticker: str, splits):
    """Populate stock_splits table"""
    try:
        cur = conn.cursor()
        count = 0
        for date, ratio in splits.items():
            if ratio != 1.0:
                cur.execute("""
                    INSERT INTO stock_splits (ticker, date, split_ratio)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (ticker, date) DO UPDATE SET split_ratio = EXCLUDED.split_ratio
                """, (ticker, date.date(), float(ratio)))
                count += 1
        conn.commit()
        return count
    except Exception as e:
        logger.error(f"Error populating stock_splits for {ticker}: {e}")
        conn.rollback()
        return 0


def main():
    """Main function to populate all tables"""
    logger.info("=" * 60)
    logger.info("Starting Complete Database Population")
    logger.info("=" * 60)
    
    conn = get_db_connection()
    logger.info(f"✅ Connected to PostgreSQL {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    # Period for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    stats = {
        'companies': 0,
        'price_history': 0,
        'dividends': 0,
        'stock_splits': 0
    }
    
    for ticker in TICKERS:
        logger.info(f"\n📊 Processing {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            
            # 1. Company info
            info = stock.info
            if populate_company(conn, ticker, info):
                stats['companies'] += 1
                logger.info(f"  ✅ Company info: {info.get('longName', ticker)}")
            
            # 2. Price history (1 year)
            hist = stock.history(period="1y")
            count = populate_price_history(conn, ticker, hist)
            stats['price_history'] += count
            logger.info(f"  ✅ Price history: {count} records")
            
            # 3. Dividends
            dividends = stock.dividends
            count = populate_dividends(conn, ticker, dividends)
            stats['dividends'] += count
            logger.info(f"  ✅ Dividends: {count} records")
            
            # 4. Stock splits
            splits = stock.splits
            count = populate_stock_splits(conn, ticker, splits)
            stats['stock_splits'] += count
            logger.info(f"  ✅ Stock splits: {count} records")
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {ticker}: {e}")
    
    conn.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("📈 Population Complete!")
    logger.info(f"  - Companies: {stats['companies']}")
    logger.info(f"  - Price History: {stats['price_history']} records")
    logger.info(f"  - Dividends: {stats['dividends']} records")
    logger.info(f"  - Stock Splits: {stats['stock_splits']} records")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
