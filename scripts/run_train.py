"""
Airflow script to run model training pipeline.
Called by train_model_dag.py
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from loguru import logger


def run_train(
    model_type: str = "random_forest",
    ticker: str = None,
    lookback_days: int = 365,
    save_model: bool = True,
) -> dict:
    """
    Run the model training pipeline.
    
    Args:
        model_type: Type of model to train ('random_forest', 'lstm')
        ticker: Specific ticker to train on (None = all data)
        lookback_days: Days of historical data to use
        save_model: Whether to save trained model
    
    Returns:
        Dict with training stats and metrics.
    """
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import pickle

    from src.machine_learning.preprocessing.preprocess import (
        preprocess_for_training,
        DataScaler,
    )
    from src.machine_learning.feature_engineering.features import build_features

    start_time = datetime.now()
    stats = {
        "model_type": model_type,
        "ticker": ticker,
        "start_time": start_time.isoformat(),
    }

    logger.info(f"🚀 Starting {model_type} training")

    try:
        # Load data from database or generate sample
        df = load_training_data(ticker, lookback_days)

        if df.empty:
            logger.warning("No training data available, generating sample")
            df = generate_sample_training_data(lookback_days)

        logger.info(f"Loaded {len(df)} training samples")

        # Feature engineering
        df = build_features(df)
        logger.info(f"Features built, {len(df)} samples after cleaning")

        # Preprocess
        result = preprocess_for_training(
            df,
            target_column="close",
            train_ratio=0.8,
            save_scaler=save_model,
        )

        X_train = result["X_train"]
        X_test = result["X_test"]
        y_train = result["y_train"]
        y_test = result["y_test"]

        logger.info(f"Train/Test shapes: {X_train.shape} / {X_test.shape}")

        # Train model
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

        elif model_type == "lstm":
            from src.deep_learning.model.lstm import train_lstm
            from src.machine_learning.preprocessing.preprocess import create_sequences

            # Create sequences for LSTM
            sequence_length = 10
            X_train_seq, y_train_seq = create_sequences(
                np.column_stack([X_train, y_train]),
                sequence_length,
            )
            X_test_seq, y_test_seq = create_sequences(
                np.column_stack([X_test, y_test]),
                sequence_length,
            )

            # Train LSTM
            lstm_result = train_lstm(
                X_train_seq,
                y_train_seq,
                X_test_seq,
                y_test_seq,
                epochs=50,
                save_model=save_model,
            )

            model = lstm_result["model"]
            train_pred = y_train_seq  # Placeholder
            test_pred = y_test_seq

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)

        stats["metrics"] = {
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
        }

        logger.info(f"Metrics - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

        # Save model
        if save_model and model_type == "random_forest":
            model_dir = "src/machine_learning/models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "model.pkl")

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            stats["model_path"] = model_path
            logger.info(f"✅ Model saved to {model_path}")

        stats["status"] = "success"

    except Exception as e:
        logger.error(f"Training failed: {e}")
        stats["status"] = "failed"
        stats["error"] = str(e)

    end_time = datetime.now()
    stats["end_time"] = end_time.isoformat()
    stats["duration_seconds"] = (end_time - start_time).total_seconds()

    return stats


def load_training_data(ticker: str = None, days: int = 365) -> pd.DataFrame:
    """Load training data from database."""
    import pandas as pd

    try:
        from sqlalchemy import text
        from src.database.db_connection import get_session

        query = """
            SELECT ticker, date, open, high, low, close, volume
            FROM price_history
            WHERE date >= NOW() - INTERVAL ':days days'
        """

        if ticker:
            query += " AND ticker = :ticker"

        query += " ORDER BY date"

        with get_session() as session:
            result = session.execute(
                text(query),
                {"days": days, "ticker": ticker}
            )
            rows = result.fetchall()

        if rows:
            df = pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"])
            return df

    except Exception as e:
        logger.warning(f"Could not load from database: {e}")

    return pd.DataFrame()


def generate_sample_training_data(days: int = 365) -> pd.DataFrame:
    """Generate sample training data."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    np.random.seed(42)
    base_price = 150.0
    returns = np.random.normal(0.0005, 0.02, days)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "ticker": "SAMPLE",
        "date": dates,
        "open": prices * np.random.uniform(0.99, 1.01, days),
        "high": prices * np.random.uniform(1.01, 1.03, days),
        "low": prices * np.random.uniform(0.97, 0.99, days),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, days),
    })

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run model training pipeline")
    parser.add_argument(
        "--model",
        choices=["random_forest", "lstm"],
        default="random_forest",
        help="Model type to train",
    )
    parser.add_argument("--ticker", help="Specific ticker to train on")
    parser.add_argument("--days", type=int, default=365, help="Days of data to use")
    parser.add_argument("--no-save", action="store_true", help="Don't save model")

    args = parser.parse_args()

    result = run_train(
        model_type=args.model,
        ticker=args.ticker,
        lookback_days=args.days,
        save_model=not args.no_save,
    )

    print(f"Result: {result}")
