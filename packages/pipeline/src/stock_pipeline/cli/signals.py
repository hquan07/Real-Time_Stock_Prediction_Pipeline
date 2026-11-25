import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from stock_pipeline.db.db_connection import get_engine
from stock_pipeline.ml.inference.inference import predict_next_price
from stock_pipeline.ml.feature_engineering.features import build_features

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
        # Read enough validated observations to calculate RSI from real prices.
        query = text("""
            SELECT ticker, event_time as date, open, high, low, close, volume
            FROM stock_prices_stream
            WHERE ticker = :ticker
            ORDER BY event_time DESC
            LIMIT 40
        """)

        with engine.connect() as conn:
            rows = conn.execute(query, {"ticker": ticker}).fetchall()

        if len(rows) < 15:
            return {}

        frame = pd.DataFrame([dict(row._mapping) for row in rows])
        frame = frame.sort_values("date")
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        featured = build_features(frame, is_training=False)
        if featured.empty:
            return {}
        latest = featured.iloc[-1].to_dict()
        latest["rsi"] = float(compute_rsi(frame["close"]).iloc[-1])
        return latest
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

        current_rsi = last_row.get("rsi")
        if current_rsi is None or pd.isna(current_rsi):
            logger.warning(f"Validated RSI unavailable for {ticker}")
            continue

        try:
            # Predict
            result = predict_next_price(ticker, last_row)

            if result.get("status") != "success":
                logger.error(f"Prediction failed for {ticker}: {result.get('error')}")
                continue

            pred_return = result.get("predicted_return", 0.0)
            # BUY/SELL publication remains disabled until backtest and paper-risk
            # gates are approved. Persist a research-only HOLD observation.
            signal_type = "HOLD"
            reason = (
                "Research only; BUY/SELL publication disabled pending backtest. "
                f"Observed RSI={current_rsi:.1f}, predicted_return={pred_return:.6f}."
            )

            signals.append({
                "ticker": ticker,
                "signal_type": signal_type,
                "reason": reason,
                "confidence": None,
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
