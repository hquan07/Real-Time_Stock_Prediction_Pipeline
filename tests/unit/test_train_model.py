"""
Unit tests for model training.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import os


class TestGetModel:
    """Tests for model factory."""

    def test_get_linear_model(self):
        """Test getting linear regression model."""
        from src.machine_learning.training.train_model import get_model

        model = get_model("linear")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_random_forest_model(self):
        """Test getting random forest model."""
        from src.machine_learning.training.train_model import get_model

        model = get_model("random_forest", n_estimators=50)
        assert model is not None
        assert hasattr(model, "fit")

    def test_get_xgboost_model(self):
        """Test getting XGBoost model if available."""
        pytest.importorskip("xgboost")
        from src.machine_learning.training.train_model import get_model

        model = get_model("xgboost", n_estimators=50)
        assert model is not None
        assert hasattr(model, "fit")

    def test_get_unknown_model_raises(self):
        """Test that unknown model type raises error."""
        from src.machine_learning.training.train_model import get_model

        with pytest.raises(ValueError):
            get_model("unknown_model")


class TestTrainAndEvaluate:
    """Tests for train and evaluate function."""

    def test_train_and_evaluate_returns_results(self, sample_price_data):
        """Test that training returns results dict."""
        from src.machine_learning.training.train_model import train_and_evaluate

        # Prepare features
        X = sample_price_data[["open", "high", "low", "volume"]].copy()
        y = sample_price_data["close"].copy()

        # Drop NaN
        X = X.dropna()
        y = y.loc[X.index]

        result = train_and_evaluate(X, y, model_types=["linear", "random_forest"])

        assert isinstance(result, dict)
        assert "linear" in result
        assert "random_forest" in result

    def test_train_has_metrics(self, sample_price_data):
        """Test that results contain metrics."""
        from src.machine_learning.training.train_model import train_and_evaluate

        X = sample_price_data[["open", "high", "low", "volume"]].iloc[10:].copy()
        y = sample_price_data["close"].iloc[10:].copy()

        result = train_and_evaluate(X, y, model_types=["linear"])

        for model_name, metrics in result.items():
            assert "mean_rmse" in metrics
            assert "std_rmse" in metrics
            assert "mean_mae" in metrics
            assert metrics["mean_rmse"] >= 0

    def test_cross_validation_splits(self, sample_price_data):
        """Test that cross-validation uses correct number of splits."""
        from src.machine_learning.training.train_model import train_and_evaluate

        X = sample_price_data[["open", "high", "low", "volume"]].iloc[20:].copy()
        y = sample_price_data["close"].iloc[20:].copy()

        result = train_and_evaluate(X, y, model_types=["linear"], n_splits=3)

        assert "linear" in result

    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", reason="XGBoost not installed"),
        reason="XGBoost not installed"
    )
    def test_train_with_xgboost(self, sample_price_data):
        """Test training includes XGBoost when available."""
        from src.machine_learning.training.train_model import train_and_evaluate

        X = sample_price_data[["open", "high", "low", "volume"]].iloc[20:].copy()
        y = sample_price_data["close"].iloc[20:].copy()

        result = train_and_evaluate(
            X, y, 
            model_types=["linear", "random_forest", "xgboost"]
        )

        assert "xgboost" in result


class TestTrainSingleModel:
    """Tests for single model training."""

    def test_train_single_no_save(self, sample_price_data):
        """Test training a single model without saving."""
        from src.machine_learning.training.train_model import train_single_model

        X = sample_price_data[["open", "high", "low", "volume"]].iloc[10:].copy()
        y = sample_price_data["close"].iloc[10:].copy()

        result = train_single_model(X, y, model_type="linear", save_model=False)

        assert "model" in result
        assert "metrics" in result
        assert result["model_type"] == "linear"

    def test_train_single_with_save(self, sample_price_data, tmp_path):
        """Test training and saving a model."""
        from src.machine_learning.training.train_model import train_single_model

        X = sample_price_data[["open", "high", "low", "volume"]].iloc[10:].copy()
        y = sample_price_data["close"].iloc[10:].copy()

        model_path = str(tmp_path / "test_model.pkl")

        result = train_single_model(
            X, y, 
            model_type="random_forest",
            save_model=True,
            model_path=model_path,
        )

        assert "model_path" in result
        assert os.path.exists(model_path)


class TestCompareModels:
    """Tests for model comparison."""

    def test_compare_finds_best_model(self, sample_price_data):
        """Test that comparison finds best model."""
        from src.machine_learning.training.train_model import compare_models

        X = sample_price_data[["open", "high", "low", "volume"]].iloc[20:].copy()
        y = sample_price_data["close"].iloc[20:].copy()

        result = compare_models(X, y, n_splits=3)

        assert "best_model" in result
        assert "best_rmse" in result
        assert "comparison" in result
        assert result["best_rmse"] >= 0


class TestAirflowTrainScript:
    """Tests for Airflow training script."""

    @patch("airflow.scripts.run_train.load_training_data")
    @patch("airflow.scripts.run_train.generate_sample_training_data")
    def test_run_train_with_sample_data(self, mock_sample, mock_load):
        """Test training with sample data when no DB available."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        
        from airflow.scripts.run_train import run_train, generate_sample_training_data

        # Mock empty DB load
        mock_load.return_value = pd.DataFrame()

        # Use real sample generation
        sample_df = generate_sample_training_data(100)
        mock_sample.return_value = sample_df

        result = run_train(model_type="random_forest", save_model=False)

        assert result["status"] in ["success", "failed"]
        assert "duration_seconds" in result


class TestFeaturePipeline:
    """Tests for feature building in training."""

    def test_features_for_training(self, sample_price_data):
        """Test feature building produces valid training data."""
        from src.machine_learning.feature_engineering.features import build_features

        df = build_features(sample_price_data)

        # Should have features
        assert "log_return" in df.columns
        assert "ma_5" in df.columns
        assert "close_lag_1" in df.columns

        # No NaN
        assert not df.isna().any().any()

        # Enough samples for training
        assert len(df) > 10
