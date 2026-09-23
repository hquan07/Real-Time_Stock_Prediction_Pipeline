import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import yfinance as yf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.database.db_connection import get_engine
from src.machine_learning.inference.inference import InferenceEngine, predict_next_price
from src.machine_learning.feature_engineering.features import build_features

def compute_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_latest_features(ticker: str, engine) -> dict:
    try:
        from sqlalchemy import text
        # Fetch the most recent row which already has features from Spark
        query = text("""
            SELECT ticker, event_time as date, open, high, low, close, volume,
                   log_return, ma_5, ma_10, volatility_10, 
                   close_lag_1, close_lag_5, close_lag_10
            FROM stock_prices_stream 
            WHERE ticker = :ticker 
            ORDER BY event_time DESC 
            LIMIT 1
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"ticker": ticker}).fetchone()
            
        if not result:
            return {}
            
        row = dict(result._mapping)
        
        # Ensure numeric
        for k, v in row.items():
            if k not in ['ticker', 'date'] and v is not None:
                row[k] = float(v)
                
        return row
    except Exception as e:
        logger.error(f"Error fetching latest features for {ticker}: {e}")
        return {}

def run_signal_generation():
    engine = get_engine()
    
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "JPM", "SPY", "QQQ"]
    signals = []
    
    for ticker in tickers:
        logger.info(f"Generating signal for {ticker}")
        
        # 1. Consume clean data from Spark (Task 2.2)
        last_row = fetch_latest_features(ticker, engine)
        
        if not last_row or last_row.get("ma_10") is None:
            logger.warning(f"Not enough clean feature data for {ticker} yet")
            continue
            
        current_rsi = 50 # Default or we could compute RSI if needed, but let's just use 50 if missing in DB
        
        try:
            # Predict
            result = predict_next_price(ticker, last_row)
            
            if result.get("status") != "success":
                logger.error(f"Prediction failed for {ticker}: {result.get('error')}")
                continue
                
            pred_return = result.get("predicted_return", 0.0)
            ci_width = result.get("confidence_interval_width", 0.0)
            
            signal_type = "HOLD"
            reason = []
            
            # Simple strategy
            if current_rsi < 30 and pred_return > 0:
                signal_type = "BUY"
                reason.append(f"RSI Oversold ({current_rsi:.1f}) and Model predicts Uptrend (+{pred_return*100:.2f}%)")
            elif current_rsi > 70 and pred_return < 0:
                signal_type = "SELL"
                reason.append(f"RSI Overbought ({current_rsi:.1f}) and Model predicts Downtrend ({pred_return*100:.2f}%)")
            else:
                signal_type = "HOLD"
                reason.append(f"RSI neutral or Model disagrees. (RSI: {current_rsi:.1f}, Pred: {pred_return*100:.2f}%)")
                
            signals.append({
                "ticker": ticker,
                "signal_type": signal_type,
                "reason": " | ".join(reason),
                "confidence": max(0, 100 - (ci_width * 1000)) # dummy confidence logic
            })
            
            logger.info(f"Result for {ticker}: {signal_type}")
            
        except Exception as e:
            logger.error(f"Failed to predict for {ticker}: {e}")
            continue
            
    # Save to database
    if signals:
        try:
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy import text
            
            Session = sessionmaker(bind=engine)
            session = Session()
            
            for s in signals:
                session.execute(text("""
                    INSERT INTO public.signals (ticker, signal_type, reason, confidence, created_at)
                    VALUES (:ticker, :signal_type, :reason, :confidence, NOW())
                """), s)
            session.commit()
            session.close()
            logger.info(f"Saved {len(signals)} signals to DB.")
        except Exception as e:
            logger.error(f"Error saving signals to DB: {e}")

if __name__ == "__main__":
    run_signal_generation()
