from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
import pandas as pd
from loguru import logger
from core.database import get_db_connection
from services.data_service import fetch_stock_data

router = APIRouter()

@router.get("/tickers")
def get_active_tickers():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        query = "SELECT ticker, long_name FROM companies WHERE is_active = true ORDER BY ticker ASC"
        df = pd.read_sql_query(query, conn)
        return [{"label": f"{row['long_name']} ({row['ticker']})", "value": row["ticker"]} for _, row in df.iterrows()]
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        return [{"label": "Apple Inc. (AAPL)", "value": "AAPL"}]
    finally:
        conn.close()

@router.get("/stock/{ticker}")
def get_stock_data(ticker: str, period: str = Query("3M", alias="period")):
    df = fetch_stock_data(ticker, period)
    if df.empty:
        return []
    
    # Convert dates to string for JSON serialization
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)
        
    return df.to_dict(orient="records")

@router.get("/portfolio/{user_id}")
def get_portfolio(user_id: str = "default_user"):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        query = """
            WITH LatestPrices AS (
                SELECT ticker, close as current_price
                FROM (
                    SELECT ticker, close, ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY event_time DESC) as rn
                    FROM stock_prices_stream
                ) tmp
                WHERE rn = 1
            )
            SELECT 
                p.ticker,
                p.shares as quantity,
                p.avg_price as average_price,
                COALESCE(lp.current_price, p.avg_price) as current_price
            FROM portfolio p
            LEFT JOIN LatestPrices lp ON p.ticker = lp.ticker
            WHERE p.user_id = %s
        """
        df = pd.read_sql_query(query, conn, params=(user_id,))
        if not df.empty:
            df["quantity"] = pd.to_numeric(df["quantity"], errors='coerce')
            df["average_price"] = pd.to_numeric(df["average_price"], errors='coerce')
            df["current_price"] = pd.to_numeric(df["current_price"], errors='coerce')
            df["total_cost"] = df["quantity"] * df["average_price"]
            df["total_value"] = df["quantity"] * df["current_price"]
            df["unrealized_pnl"] = df["total_value"] - df["total_cost"]
            df.fillna(0, inplace=True)
            return df.to_dict(orient="records")
        return []
    except Exception as e:
        logger.error(f"Failed to fetch portfolio: {e}")
        return []
    finally:
        conn.close()

@router.get("/predictions/{ticker}")
def get_predictions(ticker: str, model: str = Query("RandomForest")):
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        query = """
            SELECT prediction_date, predicted_price as predicted_close, actual_price as actual_close
            FROM predictions 
            WHERE ticker = %s AND model_type = %s
            ORDER BY prediction_date ASC
        """
        df = pd.read_sql_query(query, conn, params=(ticker, model))
        if df.empty:
            return {"historical": [], "future": []}
            
        df['confidence_upper'] = df['predicted_close'] * 1.05
        df['confidence_lower'] = df['predicted_close'] * 0.95
            
        df['prediction_date'] = df['prediction_date'].astype(str)
        df.fillna(0, inplace=True)
        return {"historical": df.to_dict(orient="records"), "future": []} # Simplified for now
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
        return {}
    finally:
        conn.close()
