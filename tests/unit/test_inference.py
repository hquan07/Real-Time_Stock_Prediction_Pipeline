"""
Unit tests for ML inference module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestModelLoader:
    """Tests for ModelLoader class."""

    @patch("builtins.open", create=True)
    @patch("pickle.load")
    def test_load_model_success(self, mock_pickle, mock_open):
        """Test successful model loading."""
        from src.machine_learning.inference.inference import ModelLoader

        mock_model = MagicMock()
        mock_pickle.return_value = mock_model

        model = ModelLoader.load_model("/path/to/model.pkl", use_cache=False)

        assert model == mock_model

    def test_load_model_file_not_found(self):
        """Test handling of missing model file."""
        from src.machine_learning.inference.inference import ModelLoader

        with pytest.raises(FileNotFoundError):
            ModelLoader.load_model("/nonexistent/model.pkl", use_cache=False)

    def test_cache_functionality(self):
        """Test that models are cached."""
        from src.machine_learning.inference.inference import ModelLoader

        ModelLoader.clear_cache()

        with patch("builtins.open", create=True):
            with patch("pickle.load") as mock_pickle:
                mock_pickle.return_value = MagicMock()

                # First load
                ModelLoader.load_model("/test/model.pkl", use_cache=True)

                # Second load should use cache
                ModelLoader.load_model("/test/model.pkl", use_cache=True)

                # Only called once
                assert mock_pickle.call_count == 1


class TestInferenceEngine:
    """Tests for InferenceEngine class."""

    @patch("src.machine_learning.inference.inference.ModelLoader.load_model")
    @patch("src.machine_learning.inference.inference.ModelLoader.load_scaler")
    def test_predict_dataframe(self, mock_scaler, mock_model):
        """Test prediction with DataFrame input."""
        from src.machine_learning.inference.inference import InferenceEngine

        # Setup mocks
        model = MagicMock()
        model.predict.return_value = np.array([150.0, 155.0])
        mock_model.return_value = model
        mock_scaler.return_value = None

        engine = InferenceEngine(feature_columns=["open", "high", "low", "close"])

        df = pd.DataFrame({
            "open": [100.0, 105.0],
            "high": [110.0, 115.0],
            "low": [95.0, 100.0],
            "close": [105.0, 110.0],
        })

        predictions = engine.predict(df)

        assert len(predictions) == 2
        assert predictions[0] == 150.0

    @patch("src.machine_learning.inference.inference.ModelLoader.load_model")
    @patch("src.machine_learning.inference.inference.ModelLoader.load_scaler")
    def test_predict_single(self, mock_scaler, mock_model):
        """Test single prediction from dict."""
        from src.machine_learning.inference.inference import InferenceEngine

        model = MagicMock()
        model.predict.return_value = np.array([155.0])
        mock_model.return_value = model
        mock_scaler.return_value = None

        engine = InferenceEngine(feature_columns=["open", "high", "low", "close"])

        result = engine.predict_single({
            "open": 100.0,
            "high": 110.0,
            "low": 95.0,
            "close": 105.0,
        })

        assert result == 155.0


class TestBatchPredict:
    """Tests for batch prediction function."""

    @patch("src.machine_learning.inference.inference.InferenceEngine")
    def test_batch_predict_adds_column(self, mock_engine_class):
        """Test that predictions are added as column."""
        from src.machine_learning.inference.inference import batch_predict

        engine = MagicMock()
        engine.predict.return_value = np.array([155.0, 160.0, 165.0])
        mock_engine_class.return_value = engine

        df = pd.DataFrame({
            "open": [100.0, 105.0, 110.0],
            "close": [105.0, 110.0, 115.0],
        })

        result = batch_predict(df, output_column="predicted")

        assert "predicted" in result.columns
        assert len(result["predicted"]) == 3


class TestGetModelInfo:
    """Tests for model info function."""

    @patch("os.path.exists")
    def test_model_not_exists(self, mock_exists):
        """Test when model doesn't exist."""
        from src.machine_learning.inference.inference import get_model_info

        mock_exists.return_value = False

        info = get_model_info("/nonexistent/model.pkl")

        assert info["exists"] is False

    @patch("os.path.exists")
    @patch("os.stat")
    @patch("src.machine_learning.inference.inference.ModelLoader.load_model")
    @patch("src.machine_learning.inference.inference.ModelLoader.load_metrics")
    def test_model_exists(self, mock_metrics, mock_load, mock_stat, mock_exists):
        """Test when model exists."""
        from src.machine_learning.inference.inference import get_model_info

        mock_exists.return_value = True
        mock_stat.return_value = MagicMock(st_size=1024, st_mtime=1704067200)
        mock_load.return_value = MagicMock(__class__=MagicMock(__name__="RandomForest"))
        mock_metrics.return_value = {"rmse": 0.05}

        info = get_model_info("/test/model.pkl")

        assert info["exists"] is True
        assert "file_size_mb" in info
