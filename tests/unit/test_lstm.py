"""
Unit tests for LSTM model.
"""

import pytest
import numpy as np


class TestLSTMModel:
    """Tests for LSTM model architecture."""

    @pytest.fixture
    def sample_sequences(self):
        """Generate sample sequence data."""
        np.random.seed(42)
        X = np.random.rand(100, 10, 5)  # 100 samples, 10 timesteps, 5 features
        y = np.random.rand(100)
        return X, y

    def test_lstm_import(self):
        """Test that LSTM module can be imported."""
        try:
            from src.deep_learning.model.lstm import LSTMModel, TORCH_AVAILABLE
            assert True
        except ImportError:
            pytest.skip("PyTorch not installed")

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch not installed"
    )
    def test_lstm_forward(self, sample_sequences):
        """Test LSTM forward pass."""
        try:
            import torch
            from src.deep_learning.model.lstm import LSTMModel

            X, y = sample_sequences

            model = LSTMModel(
                input_size=5,
                hidden_size=32,
                num_layers=2,
            )

            X_tensor = torch.FloatTensor(X[:10])
            output = model(X_tensor)

            assert output.shape == (10,)

        except ImportError:
            pytest.skip("PyTorch not installed")

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch not installed"
    )
    def test_lstm_training(self, sample_sequences):
        """Test LSTM training loop."""
        try:
            from src.deep_learning.model.lstm import train_lstm

            X, y = sample_sequences

            # Small training run
            result = train_lstm(
                X[:50],
                y[:50],
                epochs=2,
                batch_size=16,
                save_model=False,
            )

            assert "model" in result
            assert "history" in result
            assert "metrics" in result
            assert result["metrics"]["epochs_trained"] == 2

        except ImportError:
            pytest.skip("PyTorch not installed")


class TestStockDataset:
    """Tests for StockDataset class."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch not installed"
    )
    def test_dataset_length(self):
        """Test dataset length."""
        try:
            from src.deep_learning.model.lstm import StockDataset

            X = np.random.rand(100, 10, 5)
            y = np.random.rand(100)

            dataset = StockDataset(X, y)

            assert len(dataset) == 100

        except ImportError:
            pytest.skip("PyTorch not installed")

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch not installed"
    )
    def test_dataset_getitem(self):
        """Test dataset indexing."""
        try:
            import torch
            from src.deep_learning.model.lstm import StockDataset

            X = np.random.rand(100, 10, 5)
            y = np.random.rand(100)

            dataset = StockDataset(X, y)
            sample_x, sample_y = dataset[0]

            assert isinstance(sample_x, torch.Tensor)
            assert sample_x.shape == (10, 5)

        except ImportError:
            pytest.skip("PyTorch not installed")
