"""
Integration tests for training pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import tempfile


@pytest.mark.integration
class TestTrainingPipelineIntegration:
    """End-to-end training pipeline tests."""

    def test_full_training_pipeline(self, sample_price_data):
        """Test complete training pipeline from data to model."""
        from src.machine_learning.feature_engineering.features import build_features
        from src.machine_learning.preprocessing.preprocess import preprocess_for_training
        from src.machine_learning.training.train_model import train_and_evaluate

        # Step 1: Feature engineering
        df = build_features(sample_price_data)
        assert not df.empty
        assert "log_return" in df.columns

        # Step 2: Preprocessing
        result = preprocess_for_training(
            df,
            target_column="close",
            train_ratio=0.8,
            scale=True,
            save_scaler=False,
        )

        assert result["X_train"].shape[0] > 0
        assert result["y_train"].shape[0] > 0

        # Step 3: Training
        X_train = result["X_train"]
        y_train = result["y_train"]

        # Create DataFrame for train_and_evaluate
        train_df = pd.DataFrame(X_train, columns=result["feature_names"])
        train_df["close"] = y_train

        metrics = train_and_evaluate(
            train_df[result["feature_names"]],
            train_df["close"],
        )

        assert "LinearRegression" in metrics
        assert metrics["LinearRegression"]["mean_rmse"] >= 0

    def test_model_save_and_load(self, sample_price_data):
        """Test model can be saved and loaded."""
        import pickle
        from sklearn.ensemble import RandomForestRegressor
        from src.machine_learning.feature_engineering.features import build_features

        # Prepare data
        df = build_features(sample_price_data)
        X = df[["log_return", "ma_5", "ma_10"]].values
        y = df["close"].values

        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            temp_path = f.name

        try:
            # Load model
            from src.machine_learning.inference.inference import ModelLoader

            loaded_model = ModelLoader.load_model(temp_path, use_cache=False)

            # Verify predictions match
            original_pred = model.predict(X[:5])
            loaded_pred = loaded_model.predict(X[:5])

            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

        finally:
            os.unlink(temp_path)


@pytest.mark.integration
class TestInferencePipeline:
    """Integration tests for inference pipeline."""

    def test_batch_prediction(self, sample_price_data):
        """Test batch prediction workflow."""
        from src.machine_learning.feature_engineering.features import build_features
        from src.machine_learning.preprocessing.preprocess import DataScaler
        from sklearn.ensemble import RandomForestRegressor
        import pickle
        import tempfile

        # Prepare data
        df = build_features(sample_price_data)
        feature_cols = ["log_return", "ma_5", "ma_10"]
        X = df[feature_cols].values
        y = df["close"].values

        # Train and save model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            model_path = f.name

        try:
            from src.machine_learning.inference.inference import batch_predict

            # Prepare test data
            test_df = df[feature_cols].head(10).copy()

            # Run batch prediction
            result = batch_predict(
                test_df,
                model_path=model_path,
                feature_columns=feature_cols,
            )

            assert "prediction" in result.columns
            assert len(result) == 10

        finally:
            os.unlink(model_path)


@pytest.mark.integration
@pytest.mark.requires_db
class TestDatabaseIntegration:
    """Integration tests requiring database."""

    @pytest.mark.skip(reason="Requires running PostgreSQL")
    def test_write_and_read_predictions(self):
        """Test writing and reading predictions from database."""
        from src.database.write_prediction import (
            write_prediction,
            get_latest_predictions,
            ensure_predictions_table,
        )
        from datetime import date

        # Ensure table exists
        ensure_predictions_table()

        # Write prediction
        write_prediction(
            ticker="TEST",
            prediction_date=date.today(),
            target_date=date.today(),
            model_name="pytest",
            predicted_close=100.0,
        )

        # Read back
        predictions = get_latest_predictions("TEST", limit=1)

        assert len(predictions) == 1
        assert predictions[0]["ticker"] == "TEST"
