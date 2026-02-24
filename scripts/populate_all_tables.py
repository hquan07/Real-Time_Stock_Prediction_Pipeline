#!/usr/bin/env python3
"""
Complete Database Populator - All Tables (Fixed Schema)
"""

import yfinance as yf
import psycopg2
from datetime import datetime
from loguru import logger
import pandas as pd

# PostgreSQL config  
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "stockdb"
DB_USER = "postgres"
DB_PASSWORD = "Huyquan1607"

TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']


def get_db_connection():
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)


def safe_float(value):
    try:
        if pd.isna(value) or value is None:
            return None
        return float(value)
    except:
        return None


def safe_int(value):
    try:
        if pd.isna(value) or value is None:
            return None
        return int(value)
    except:
        return None


def populate_key_metrics(conn, ticker: str, info: dict):
    """key_metrics: ticker, date, market_cap, trailing_pe, forward_pe, peg_ratio, beta, roa, roe, profit_margin, debt_to_equity, current_ratio"""
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO key_metrics (ticker, date, market_cap, trailing_pe, forward_pe, peg_ratio, beta, roa, roe, profit_margin, debt_to_equity, current_ratio)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, date) DO UPDATE SET market_cap = EXCLUDED.market_cap
        """, (
            ticker, datetime.now().date(),
            safe_int(info.get('marketCap')),
            safe_float(info.get('trailingPE')),
            safe_float(info.get('forwardPE')),
            safe_float(info.get('pegRatio')),
            safe_float(info.get('beta')),
            safe_float(info.get('returnOnAssets')),
            safe_float(info.get('returnOnEquity')),
            safe_float(info.get('profitMargins')),
            safe_float(info.get('debtToEquity')),
            safe_float(info.get('currentRatio'))
        ))
        conn.commit()
        return 1
    except Exception as e:
        logger.error(f"key_metrics {ticker}: {e}")
        conn.rollback()
        return 0


def populate_earnings_history(conn, ticker: str, stock):
    """earnings_history: ticker, period, revenue_actual, earnings_actual, earnings_estimate"""
    try:
        earnings = stock.earnings_history
        if earnings is None or earnings.empty:
            return 0
        cur = conn.cursor()
        count = 0
        for _, row in earnings.iterrows():
            period = row.get('Earnings Date')
            if hasattr(period, 'date'):
                period = period.date()
            else:
                period = datetime.now().date()
            cur.execute("""
                INSERT INTO earnings_history (ticker, period, earnings_actual, earnings_estimate)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ticker, period) DO NOTHING
            """, (ticker, period, safe_float(row.get('Reported EPS')), safe_float(row.get('EPS Estimate'))))
            count += 1
        conn.commit()
        return count
    except Exception as e:
        logger.debug(f"earnings_history {ticker}: {e}")
        conn.rollback()
        return 0


def populate_analyst_recommendations(conn, ticker: str, stock):
    """analyst_recommendations: ticker, date, firm, recommendation, target_price"""
    try:
        recs = stock.recommendations
        if recs is None or recs.empty:
            return 0
        cur = conn.cursor()
        count = 0
        for date, row in recs.tail(20).iterrows():
            rec_date = date.date() if hasattr(date, 'date') else datetime.now().date()
            cur.execute("""
                INSERT INTO analyst_recommendations (ticker, date, firm, recommendation, target_price)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (ticker, rec_date, str(row.get('Firm', ''))[:100], str(row.get('To Grade', ''))[:50], None))
            count += 1
        conn.commit()
        return count
    except Exception as e:
        logger.debug(f"analyst_recommendations {ticker}: {e}")
        conn.rollback()
        return 0


def populate_major_holders(conn, ticker: str, stock):
    """major_holders: ticker, holder_type, percentage"""
    try:
        holders = stock.major_holders
        if holders is None or holders.empty:
            return 0
        cur = conn.cursor()
        count = 0
        for _, row in holders.iterrows():
            cur.execute("""
                INSERT INTO major_holders (ticker, holder_type, percentage)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (ticker, str(row.iloc[1])[:100] if len(row) > 1 else 'Unknown',
                  safe_float(str(row.iloc[0]).replace('%', '')) if len(row) > 0 else None))
            count += 1
        conn.commit()
        return count
    except Exception as e:
        logger.debug(f"major_holders {ticker}: {e}")
        conn.rollback()
        return 0


def populate_institutional_holders(conn, ticker: str, stock):
    """institutional_holders: ticker, holder, shares, date_reported"""
    try:
        inst = stock.institutional_holders
        if inst is None or inst.empty:
            return 0
        cur = conn.cursor()
        count = 0
        for _, row in inst.iterrows():
            date_rep = row.get('Date Reported')
            if hasattr(date_rep, 'date'):
                date_rep = date_rep.date()
            else:
                date_rep = None
            cur.execute("""
                INSERT INTO institutional_holders (ticker, holder, shares, date_reported)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ticker, holder) DO UPDATE SET shares = EXCLUDED.shares
            """, (ticker, str(row.get('Holder', ''))[:200], safe_int(row.get('Shares')), date_rep))
            count += 1
        conn.commit()
        return count
    except Exception as e:
        logger.debug(f"institutional_holders {ticker}: {e}")
        conn.rollback()
        return 0


def populate_income_statements(conn, ticker: str, stock):
    """income_statements: ticker, period, total_revenue, gross_profit, ..."""
    try:
        income = stock.income_stmt
        if income is None or income.empty:
            return 0
        cur = conn.cursor()
        count = 0
        for date in income.columns:
            data = income[date]
            period = date.date() if hasattr(date, 'date') else date
            cur.execute("""
                INSERT INTO income_statements (ticker, period, total_revenue, gross_profit)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ticker, period) DO UPDATE SET total_revenue = EXCLUDED.total_revenue
            """, (ticker, period, safe_float(data.get('Total Revenue')), safe_float(data.get('Gross Profit'))))
            count += 1
        conn.commit()
        return count
    except Exception as e:
        logger.debug(f"income_statements {ticker}: {e}")
        conn.rollback()
        return 0


def populate_balance_sheets(conn, ticker: str, stock):
    """balance_sheets: ticker, period, total_assets, total_liabilities, total_equity, cash, long_term_debt"""
    try:
        bs = stock.balance_sheet
        if bs is None or bs.empty:
            return 0
        cur = conn.cursor()
        count = 0
        for date in bs.columns:
            data = bs[date]
            period = date.date() if hasattr(date, 'date') else date
            cur.execute("""
                INSERT INTO balance_sheets (ticker, period, total_assets, total_liabilities, total_equity, cash, long_term_debt)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, period) DO UPDATE SET total_assets = EXCLUDED.total_assets
            """, (ticker, period,
                  safe_float(data.get('Total Assets')),
                  safe_float(data.get('Total Liabilities Net Minority Interest')),
                  safe_float(data.get('Stockholders Equity')),
                  safe_float(data.get('Cash And Cash Equivalents')),
                  safe_float(data.get('Long Term Debt'))))
            count += 1
        conn.commit()
        return count
    except Exception as e:
        logger.debug(f"balance_sheets {ticker}: {e}")
        conn.rollback()
        return 0


def populate_cashflow_statements(conn, ticker: str, stock):
    """cashflow_statements: ticker, period, operating_cashflow, investing_cashflow, financing_cashflow, free_cashflow"""
    try:
        cf = stock.cashflow
        if cf is None or cf.empty:
            return 0
        cur = conn.cursor()
        count = 0
        for date in cf.columns:
            data = cf[date]
            period = date.date() if hasattr(date, 'date') else date
            cur.execute("""
                INSERT INTO cashflow_statements (ticker, period, operating_cashflow, investing_cashflow, financing_cashflow, free_cashflow)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, period) DO UPDATE SET operating_cashflow = EXCLUDED.operating_cashflow
            """, (ticker, period,
                  safe_float(data.get('Operating Cash Flow')),
                  safe_float(data.get('Investing Cash Flow')),
                  safe_float(data.get('Financing Cash Flow')),
                  safe_float(data.get('Free Cash Flow'))))
            count += 1
        conn.commit()
        return count
    except Exception as e:
        logger.debug(f"cashflow_statements {ticker}: {e}")
        conn.rollback()
        return 0


def populate_earnings_calendar(conn, ticker: str, stock):
    """earnings_calendar: ticker, date, eps_estimate, revenue_estimate"""
    try:
        cal = stock.calendar
        if cal is None:
            return 0
        cur = conn.cursor()
        if isinstance(cal, dict):
            earnings_date = cal.get('Earnings Date')
            if earnings_date:
                dates = earnings_date if isinstance(earnings_date, list) else [earnings_date]
                for ed in dates:
                    ed_date = ed.date() if hasattr(ed, 'date') else ed
                    cur.execute("""
                        INSERT INTO earnings_calendar (ticker, date, eps_estimate, revenue_estimate)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (ticker, date) DO NOTHING
                    """, (ticker, ed_date, safe_float(cal.get('EPS Estimate')), safe_float(cal.get('Revenue Estimate'))))
        conn.commit()
        return 1
    except Exception as e:
        logger.debug(f"earnings_calendar {ticker}: {e}")
        conn.rollback()
        return 0


def main():
    logger.info("=" * 60)
    logger.info("Populating All Database Tables (Fixed Schema)")
    logger.info("=" * 60)
    
    conn = get_db_connection()
    logger.info(f"✅ Connected to PostgreSQL")
    
    stats = {'key_metrics': 0, 'earnings_history': 0, 'analyst_recommendations': 0, 
             'major_holders': 0, 'institutional_holders': 0, 'income_statements': 0,
             'balance_sheets': 0, 'cashflow_statements': 0, 'earnings_calendar': 0}
    
    for ticker in TICKERS:
        logger.info(f"\n📊 Processing {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            stats['key_metrics'] += populate_key_metrics(conn, ticker, info)
            stats['earnings_history'] += populate_earnings_history(conn, ticker, stock)
            stats['analyst_recommendations'] += populate_analyst_recommendations(conn, ticker, stock)
            stats['major_holders'] += populate_major_holders(conn, ticker, stock)
            stats['institutional_holders'] += populate_institutional_holders(conn, ticker, stock)
            stats['income_statements'] += populate_income_statements(conn, ticker, stock)
            stats['balance_sheets'] += populate_balance_sheets(conn, ticker, stock)
            stats['cashflow_statements'] += populate_cashflow_statements(conn, ticker, stock)
            stats['earnings_calendar'] += populate_earnings_calendar(conn, ticker, stock)
            
            logger.info(f"  ✅ {ticker} completed")
        except Exception as e:
            logger.error(f"  ❌ {ticker}: {e}")
    
    conn.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("📈 Population Complete!")
    for table, count in stats.items():
        logger.info(f"  - {table}: {count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
