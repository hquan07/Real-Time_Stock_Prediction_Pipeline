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

def fetch_data(ticker: str, engine, lookback_days: int = 60) -> pd.DataFrame:
    try:
        from sqlalchemy import text
        query = text("""
            SELECT ticker, date, open, high, low, close, volume 
            FROM price_history 
            WHERE ticker = :ticker 
            ORDER BY date DESC 
            LIMIT :limit
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"ticker": ticker, "limit": lookback_days})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        if df.empty:
            logger.warning(f"No data for {ticker} in DB. Falling back to yfinance.")
            hist = yf.download(ticker, period=f"{lookback_days}d", progress=False)
            if hist.empty:
                return pd.DataFrame()
                
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist = hist.reset_index()
            hist.columns = [c.lower() for c in hist.columns]
            if 'date' not in hist.columns:
                hist.rename(columns={'datetime': 'date'}, inplace=True)
            hist['ticker'] = ticker
            
            req_cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
            df = hist[[c for c in req_cols if c in hist.columns]].copy()
            for c in req_cols:
                if c not in df.columns:
                    df[c] = 0
            df = df[req_cols]
            
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def run_signal_generation():
    engine = get_engine()
    
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "JPM", "SPY", "QQQ"]
    signals = []
    
    for ticker in tickers:
        logger.info(f"Generating signal for {ticker}")
        df = fetch_data(ticker, engine, lookback_days=100)
        
        if df.empty or len(df) < 30:
            logger.warning(f"Not enough data for {ticker}")
            continue
            
        # Ensure numeric
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
                
        # Compute RSI
        df["rsi"] = compute_rsi(df["close"])
        current_rsi = df["rsi"].iloc[-1]
        
        # Build features for inference
        features_df = build_features(df.copy())
        if features_df.empty:
            continue
            
        last_row = features_df.iloc[-1:].to_dict(orient='records')[0]
        
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
