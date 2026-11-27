"""
Module to write ML/DL predictions to PostgreSQL.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional

from sqlalchemy import text
from loguru import logger

from .db_connection import get_session, get_engine


def ensure_predictions_table() -> bool:
    """Fail closed when versioned migrations have not created predictions."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT to_regclass('public.predictions') IS NOT NULL")
            ).scalar()
        if not exists:
            raise RuntimeError("predictions table missing; run scripts/migrate.py up")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create predictions table: {e}")
        return False


# ---------------------------------------------------------------------------
# Write Single Prediction
# ---------------------------------------------------------------------------
def write_prediction(
    ticker: str,
    prediction_date: date,
    target_date: date,
    model_name: str,
    predicted_close: float = None,
    predicted_return: float = None,
    confidence: float = None,
    uncertainty_validated: bool = False,
    model_version: str = None,
    features_used: dict = None,
) -> bool:
    """
    Write a single prediction to the database.
    
    Args:
        ticker: Stock ticker symbol
        prediction_date: Date when prediction was made
        target_date: Date the prediction is for
        model_name: Name of the model (e.g., "RandomForest", "LSTM")
        predicted_close: Predicted closing price
        predicted_return: Predicted return percentage
        confidence: Model confidence score (0-1)
        model_version: Version of the model
        features_used: Dict of features used for prediction
    
    Returns:
        True if successful, False otherwise.
    """
    import json

    query = text("""
        INSERT INTO predictions (
            ticker, prediction_date, target_date, model_name, model_version,
            predicted_close, predicted_return, confidence, uncertainty_validated,
            features_used
        )
        VALUES (
            :ticker, :prediction_date, :target_date, :model_name, :model_version,
            :predicted_close, :predicted_return, :confidence,
            :uncertainty_validated, :features_used
        )
        ON CONFLICT (ticker, prediction_date, target_date, model_name) DO UPDATE SET
            predicted_close = EXCLUDED.predicted_close,
            predicted_return = EXCLUDED.predicted_return,
            confidence = EXCLUDED.confidence,
            uncertainty_validated = EXCLUDED.uncertainty_validated,
            model_version = EXCLUDED.model_version,
            features_used = EXCLUDED.features_used,
            created_at = NOW()
    """)

    try:
        with get_session() as session:
            session.execute(query, {
                "ticker": ticker,
                "prediction_date": prediction_date,
                "target_date": target_date,
                "model_name": model_name,
                "model_version": model_version,
                "predicted_close": predicted_close,
                "predicted_return": predicted_return,
                "confidence": confidence,
                "uncertainty_validated": uncertainty_validated,
                "features_used": json.dumps(features_used) if features_used else None,
            })
        logger.debug(f"✅ Prediction for {ticker} (target: {target_date}) saved")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to write prediction for {ticker}: {e}")
        return False


# ---------------------------------------------------------------------------
# Write Batch Predictions
# ---------------------------------------------------------------------------
def write_batch_predictions(predictions: List[Dict[str, Any]]) -> int:
    """
    Write multiple predictions at once.
    
    Args:
        predictions: List of prediction dicts with keys:
            - ticker, prediction_date, target_date, model_name
            - Optional: predicted_close, predicted_return, confidence, 
              model_version, features_used
    
    Returns:
        Number of predictions successfully written.
    """
    import json

    query = text("""
        INSERT INTO predictions (
            ticker, prediction_date, target_date, model_name, model_version,
            predicted_close, predicted_return, confidence, uncertainty_validated,
            features_used
        )
        VALUES (
            :ticker, :prediction_date, :target_date, :model_name, :model_version,
            :predicted_close, :predicted_return, :confidence,
            :uncertainty_validated, :features_used
        )
        ON CONFLICT (ticker, prediction_date, target_date, model_name) DO UPDATE SET
            predicted_close = EXCLUDED.predicted_close,
            predicted_return = EXCLUDED.predicted_return,
            confidence = EXCLUDED.confidence,
            uncertainty_validated = EXCLUDED.uncertainty_validated,
            model_version = EXCLUDED.model_version,
            features_used = EXCLUDED.features_used,
            created_at = NOW()
    """)

    success_count = 0

    try:
        with get_session() as session:
            for pred in predictions:
                try:
                    # Prepare features_used as JSON
                    features = pred.get("features_used")
                    if features and isinstance(features, dict):
                        features = json.dumps(features)

                    session.execute(query, {
                        "ticker": pred["ticker"],
                        "prediction_date": pred["prediction_date"],
                        "target_date": pred["target_date"],
                        "model_name": pred["model_name"],
                        "model_version": pred.get("model_version"),
                        "predicted_close": pred.get("predicted_close"),
                        "predicted_return": pred.get("predicted_return"),
                        "confidence": pred.get("confidence"),
                        "uncertainty_validated": pred.get("uncertainty_validated", False),
                        "features_used": features,
                    })
                    success_count += 1

                except Exception as e:
                    logger.warning(f"⚠️ Skipping prediction: {e}")
                    continue

        logger.info(f"✅ Batch wrote {success_count}/{len(predictions)} predictions")
        return success_count

    except Exception as e:
        logger.error(f"❌ Batch prediction write failed: {e}")
        return success_count


# ---------------------------------------------------------------------------
# Read Predictions
# ---------------------------------------------------------------------------
def get_latest_predictions(
    ticker: str = None,
    model_name: str = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Retrieve latest predictions from database.
    
    Args:
        ticker: Filter by ticker (optional)
        model_name: Filter by model name (optional)
        limit: Maximum number of records to return
    
    Returns:
        List of prediction dictionaries.
    """
    base_query = """
        SELECT 
            ticker, prediction_date, target_date, model_name, model_version,
            predicted_close, predicted_return, confidence, created_at
        FROM predictions
        WHERE 1=1
    """
    params = {"limit": limit}

    if ticker:
        base_query += " AND ticker = :ticker"
        params["ticker"] = ticker

    if model_name:
        base_query += " AND model_name = :model_name"
        params["model_name"] = model_name

    base_query += " ORDER BY created_at DESC LIMIT :limit"

    try:
        with get_session() as session:
            result = session.execute(text(base_query), params)
            rows = result.fetchall()

            predictions = []
            for row in rows:
                predictions.append({
                    "ticker": row[0],
                    "prediction_date": row[1],
                    "target_date": row[2],
                    "model_name": row[3],
                    "model_version": row[4],
                    "predicted_close": float(row[5]) if row[5] else None,
                    "predicted_return": float(row[6]) if row[6] else None,
                    "confidence": float(row[7]) if row[7] else None,
                    "created_at": row[8],
                })

            return predictions

    except Exception as e:
        logger.error(f"❌ Failed to get predictions: {e}")
        return []


# ---------------------------------------------------------------------------
# Prediction Accuracy Tracking
# ---------------------------------------------------------------------------
def calculate_prediction_accuracy(
    ticker: str,
    model_name: str,
    lookback_days: int = 30,
) -> Optional[Dict[str, float]]:
    """
    Calculate prediction accuracy for a model by comparing predictions
    with actual prices.
    
    Returns:
        Dict with accuracy metrics (mape, rmse, direction_accuracy)
        or None if insufficient data.
    """
    query = text("""
        WITH prediction_vs_actual AS (
            SELECT 
                p.ticker,
                p.target_date,
                p.predicted_close,
                ph.close as actual_close
            FROM predictions p
            JOIN price_history ph 
                ON p.ticker = ph.ticker 
                AND p.target_date = ph.date
            WHERE p.ticker = :ticker
                AND p.model_name = :model_name
                AND p.created_at >= NOW() - INTERVAL ':days days'
        )
        SELECT 
            COUNT(*) as count,
            AVG(ABS(predicted_close - actual_close) / actual_close) as mape,
            SQRT(AVG(POWER(predicted_close - actual_close, 2))) as rmse
        FROM prediction_vs_actual
        WHERE actual_close IS NOT NULL AND actual_close > 0
    """)

    try:
        with get_session() as session:
            result = session.execute(query, {
                "ticker": ticker,
                "model_name": model_name,
                "days": lookback_days,
            })
            row = result.fetchone()

            if row and row[0] > 0:
                return {
                    "count": row[0],
                    "mape": float(row[1]) if row[1] else None,
                    "rmse": float(row[2]) if row[2] else None,
                }

        return None

    except Exception as e:
        logger.error(f"❌ Failed to calculate accuracy: {e}")
        return None
