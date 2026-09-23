from fastapi import APIRouter, HTTPException, Query, Request
from typing import List, Dict, Any
import pandas as pd
from loguru import logger
from services.data_service import fetch_stock_data
from services.redis_cache import get_cached_data, set_cached_data

router = APIRouter()

@router.get("/tickers")
async def get_active_tickers(request: Request):
    pool = getattr(request.app.state, "pool", None)
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection pool missing")
    
    cache_key = "api:tickers:active"
    cached = await get_cached_data(cache_key)
    if cached:
        return cached

    try:
        query = "SELECT ticker, long_name FROM companies WHERE is_active = true ORDER BY ticker ASC"
        async with pool.acquire() as conn:
            records = await conn.fetch(query)
            
        result = [{"label": f"{r['long_name']} ({r['ticker']})", "value": r["ticker"]} for r in records]
        
        # Cache for 1 hour
        await set_cached_data(cache_key, result, expire=3600)
        return result
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        return [{"label": "Apple Inc. (AAPL)", "value": "AAPL"}]

@router.get("/stock/{ticker}")
async def get_stock_data(request: Request, ticker: str, period: str = Query("3M", alias="period")):
    pool = getattr(request.app.state, "pool", None)
    cache_key = f"api:stock:{ticker}:{period}"
    
    cached = await get_cached_data(cache_key)
    if cached:
        return cached

    df = await fetch_stock_data(pool, ticker, period)
    if df.empty:
        return []
    
    # Convert dates to string for JSON serialization
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)
        
    result = df.to_dict(orient="records")
    # Cache for 5 minutes
    await set_cached_data(cache_key, result, expire=300)
    return result

@router.get("/portfolio/{user_id}")
async def get_portfolio(request: Request, user_id: str = "default_user"):
    pool = getattr(request.app.state, "pool", None)
    if not pool:
        return []
        
    cache_key = f"api:portfolio:{user_id}"
    cached = await get_cached_data(cache_key)
    if cached:
        return cached

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
            WHERE p.user_id = $1
        """
        async with pool.acquire() as conn:
            records = await conn.fetch(query, user_id)
            
        if records:
            df = pd.DataFrame([dict(r) for r in records])
            df["quantity"] = pd.to_numeric(df["quantity"], errors='coerce')
            df["average_price"] = pd.to_numeric(df["average_price"], errors='coerce')
            df["current_price"] = pd.to_numeric(df["current_price"], errors='coerce')
            df["total_cost"] = df["quantity"] * df["average_price"]
            df["total_value"] = df["quantity"] * df["current_price"]
            df["unrealized_pnl"] = df["total_value"] - df["total_cost"]
            df.fillna(0, inplace=True)
            
            result = df.to_dict(orient="records")
            # Cache for 1 minute (portfolio changes often based on real-time prices)
            await set_cached_data(cache_key, result, expire=60)
            return result
        return []
    except Exception as e:
        logger.error(f"Failed to fetch portfolio: {e}")
        return []

@router.get("/predictions/{ticker}")
async def get_predictions(request: Request, ticker: str, model: str = Query("RandomForest")):
    pool = getattr(request.app.state, "pool", None)
    if not pool:
        return {}
    
    cache_key = f"api:predictions:{ticker}:{model}"
    cached = await get_cached_data(cache_key)
    if cached:
        return cached
        
    try:
        query = """
            SELECT prediction_date, predicted_price as predicted_close, actual_price as actual_close
            FROM predictions 
            WHERE ticker = $1 AND model_type = $2
            ORDER BY prediction_date ASC
        """
        async with pool.acquire() as conn:
            records = await conn.fetch(query, ticker, model)
            
        if not records:
            return {"historical": [], "future": []}
            
        df = pd.DataFrame([dict(r) for r in records])
        df['confidence_upper'] = df['predicted_close'] * 1.05
        df['confidence_lower'] = df['predicted_close'] * 0.95
            
        df['prediction_date'] = df['prediction_date'].astype(str)
        df.fillna(0, inplace=True)
        
        result = {"historical": df.to_dict(orient="records"), "future": []}
        # Cache for 30 minutes since predictions don't change often
        await set_cached_data(cache_key, result, expire=1800)
        return result
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
        return {}
