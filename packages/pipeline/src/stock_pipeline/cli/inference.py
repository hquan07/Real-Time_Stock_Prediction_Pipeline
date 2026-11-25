"""
Airflow script to run batch inference pipeline.
Called by prediction_dag.py
"""

from datetime import datetime, date

from loguru import logger
import pandas as pd


def run_inference(
    tickers: list = None,
    model_type: str = "random_forest",
    save_predictions: bool = True,
) -> dict:
    """
    Run batch inference pipeline.

    Args:
        tickers: List of tickers to predict (None = all available)
        model_type: Model to use for predictions
        save_predictions: Whether to save predictions to database

    Returns:
        Dict with inference stats.
    """
    import pandas as pd
    import numpy as np

    from stock_pipeline.ml.inference.inference import (
        InferenceEngine,
        batch_predict,
        get_model_info,
    )
    from stock_pipeline.db.write_prediction import (
        write_prediction,
        write_batch_predictions,
        ensure_predictions_table,
    )

    start_time = datetime.now()
    stats = {
        "model_type": model_type,
        "start_time": start_time.isoformat(),
        "predictions_made": 0,
        "predictions_saved": 0,
    }

    logger.info(f"🚀 Starting batch inference with {model_type}")

    try:
        # Ensure predictions table exists
        if save_predictions and not ensure_predictions_table():
            raise RuntimeError("predictions table missing; run migrations before inference")

        # Get model info
        model_info = get_model_info()
        stats["model_info"] = model_info

        # Load latest data for prediction
        df = load_latest_data(tickers)

        if df.empty:
            raise ValueError("No validated data available; inference aborted")

        logger.info(f"Loaded {len(df)} records for inference")

        from stock_pipeline.ml.feature_engineering.features import build_features

        featured = build_features(df, is_training=False)
        if featured.empty:
            raise ValueError("Validated data cannot produce the required model features")
        featured = featured.groupby("ticker", sort=False).tail(1).reset_index(drop=True)

        # Run batch prediction
        df_predictions = batch_predict(
            featured,
            output_column="predicted_return",
        )

        # Calculate predicted close from log return
        df_predictions["predicted_close"] = df_predictions["close"] * np.exp(df_predictions["predicted_return"])

        stats["predictions_made"] = len(df_predictions)

        # Prepare predictions for database
        if save_predictions:
            predictions = []

            for _, row in df_predictions.iterrows():
                predictions.append({
                    "ticker": row.get("ticker", "UNKNOWN"),
                    "prediction_date": date.today(),
                    "target_date": date.today(),
                    "model_name": model_type,
                    "predicted_close": float(row.get("predicted_close")),
                    "predicted_return": float(row.get("predicted_return")),
                    "confidence": None,
                    "uncertainty_validated": False,
                    "model_version": model_info.get("version", "1.0"),
                })

            # Save to database
            saved_count = write_batch_predictions(predictions)
            stats["predictions_saved"] = saved_count

            logger.info(f"✅ Saved {saved_count} predictions to database")

        stats["status"] = "success"

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        stats["status"] = "failed"
        stats["error"] = str(e)

    end_time = datetime.now()
    stats["end_time"] = end_time.isoformat()
    stats["duration_seconds"] = (end_time - start_time).total_seconds()

    return stats


def load_latest_data(tickers: list = None) -> pd.DataFrame:
    """Load latest data for inference."""
    import pandas as pd

    try:
        from sqlalchemy import text
        from stock_pipeline.db.db_connection import get_session

        query = """
            SELECT ticker, event_time AS date, open, high, low, close, volume
            FROM stock_prices_stream
            WHERE event_time >= NOW() - INTERVAL '180 days'
            ORDER BY ticker, event_time
        """

        with get_session() as session:
            result = session.execute(text(query))
            rows = result.fetchall()

        if rows:
            df = pd.DataFrame(
                rows,
                columns=["ticker", "date", "open", "high", "low", "close", "volume"]
            )

            if tickers:
                df = df[df["ticker"].isin(tickers)]

            return df

    except Exception as e:
        logger.warning(f"Could not load from database: {e}")

    return pd.DataFrame()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run batch inference pipeline")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to predict")
    parser.add_argument(
        "--model",
        choices=["random_forest", "lstm"],
        default="random_forest",
        help="Model type to use",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save predictions")

    args = parser.parse_args()

    result = run_inference(
        tickers=args.tickers,
        model_type=args.model,
        save_predictions=not args.no_save,
    )

    print(f"Result: {result}")
    if result.get("status") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
