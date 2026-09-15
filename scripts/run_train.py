"""
Airflow script to run model training pipeline.
Called by train_model_dag.py
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from loguru import logger
import pandas as pd


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
            logger.warning("No training data available in the database. Falling back to yfinance for testing.")
            import yfinance as yf
            import numpy as np
            
            ticker_symbol = ticker if ticker else "AAPL"
            hist = yf.download(ticker_symbol, period=f"{lookback_days}d", progress=False)
            
            if hist.empty:
                raise ValueError(f"Failed to fetch data from yfinance for {ticker_symbol}")
                
            # Flatten multi-index columns if present (yfinance >= 0.2.0)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
                
            hist = hist.reset_index()
            # Rename columns to match database schema
            hist.columns = [c.lower() for c in hist.columns]
            if 'date' not in hist.columns:
                hist.rename(columns={'datetime': 'date'}, inplace=True)
                
            hist['ticker'] = ticker_symbol
            
            # Select required columns
            req_cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
            df = hist[[c for c in req_cols if c in hist.columns]]
            
            # Fill missing required columns
            for c in req_cols:
                if c not in df.columns:
                    df[c] = 0
            
            df = df[req_cols]

        logger.info(f"Loaded {len(df)} training samples")

        # Ensure numeric columns are float
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)

        # Feature engineering
        df = build_features(df)
        logger.info(f"Features built, {len(df)} samples after cleaning")

        # Preprocess with next-day return as target
        result = preprocess_for_training(
            df,
            target_column="target_return",
            train_ratio=0.8,
            save_scaler=False,  # We will save it via ModelRegistry
        )

        X_train = result["X_train"]
        X_test = result["X_test"]
        y_train = result["y_train"]
        y_test = result["y_test"]
        scaler = result["scaler"]
        feature_names = result["feature_names"]

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
            
            # Confidence Interval via prediction standard deviation across trees
            # For each sample, collect predictions from all trees
            preds_trees = np.stack([tree.predict(X_test) for tree in model.estimators_])
            ci_std = np.std(preds_trees, axis=0)
            
            # Example: 95% CI is approx pred +/- 1.96 * ci_std
            avg_ci_width = np.mean(ci_std) * 1.96
            stats["confidence_interval_width"] = float(avg_ci_width)

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
                save_model=False,
            )

            model = lstm_result["model"]
            train_pred = y_train_seq  # Placeholder
            test_pred = y_test_seq
            
            # Simple CI for LSTM using validation RMSE as uncertainty measure
            val_rmse = lstm_result.get("history", {}).get("val_loss", [0])[-1] ** 0.5
            stats["confidence_interval_width"] = float(val_rmse * 1.96)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)

        metrics = {
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
        }
        stats["metrics"] = metrics

        logger.info(f"Metrics - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

        # Save model using Model Registry
        if save_model:
            from src.machine_learning.model_registry import ModelRegistry
            
            # For Random Forest, we save the scikit-learn model
            # For PyTorch LSTM, ModelRegistry uses pickle by default, which works but torch.save is preferred.
            # To keep things simple we use pickle for now, or adapt later.
            version = ModelRegistry.save_model(
                model=model,
                scaler=scaler,
                features=feature_names,
                metrics=metrics,
                model_type=model_type
            )
            
            stats["model_version"] = version

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
            SELECT ticker, event_date as date, open, high, low, close, volume
            FROM stock_prices_stream
            WHERE event_date >= NOW() - (:days * INTERVAL '1 day')
        """

        if ticker:
            query += " AND ticker = :ticker"

        query += " ORDER BY event_date"

        with get_session() as session:
            result = session.execute(
                text(query),
                {"days": days, "ticker": ticker}
            )
            rows = result.fetchall()

        if rows:
            df = pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"])
            return df
        else:
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Could not load from database: {e}")
        return pd.DataFrame()


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
