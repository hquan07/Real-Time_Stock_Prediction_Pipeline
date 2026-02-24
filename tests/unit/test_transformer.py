"""
Unit tests for Transformer model.
"""

import pytest
import numpy as np

# Skip tests if PyTorch not available
torch = pytest.importorskip("torch")

from src.deep_learning.model.transformer import (
    TransformerModel,
    PositionalEncoding,
    StockDataset,
    train_transformer,
    predict_transformer,
    save_transformer_model,
    load_transformer_model,
    TORCH_AVAILABLE,
)


class TestPositionalEncoding:
    """Tests for PositionalEncoding class."""

    def test_init(self):
        """Test initialization."""
        d_model = 64
        pe = PositionalEncoding(d_model=d_model, max_len=100)
        assert pe is not None

    def test_forward_shape(self):
        """Test forward pass preserves shape."""
        d_model = 64
        batch_size = 4
        seq_len = 10

        pe = PositionalEncoding(d_model=d_model)
        x = torch.zeros(batch_size, seq_len, d_model)
        output = pe(x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformerModel:
    """Tests for TransformerModel class."""

    def test_init(self):
        """Test model initialization."""
        model = TransformerModel(
            input_size=5,
            d_model=32,
            nhead=4,
            num_encoder_layers=2,
        )
        assert model is not None
        assert model.input_size == 5
        assert model.d_model == 32

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 4
        seq_len = 10
        input_size = 5

        model = TransformerModel(
            input_size=input_size,
            d_model=32,
            nhead=4,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)

        assert output.shape == (batch_size,)

    def test_forward_with_different_seq_lengths(self):
        """Test model handles different sequence lengths."""
        model = TransformerModel(input_size=5, d_model=32, nhead=4)

        for seq_len in [5, 10, 20]:
            x = torch.randn(2, seq_len, 5)
            output = model(x)
            assert output.shape == (2,)


class TestStockDataset:
    """Tests for StockDataset class."""

    def test_init(self):
        """Test dataset initialization."""
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(100)
        dataset = StockDataset(X, y)
        assert len(dataset) == 100

    def test_getitem(self):
        """Test getting items from dataset."""
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(100)
        dataset = StockDataset(X, y)

        x_sample, y_sample = dataset[0]
        assert x_sample.shape == (10, 5)
        assert y_sample.shape == ()


class TestTrainTransformer:
    """Tests for training functions."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 50
        seq_len = 10
        n_features = 5

        X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        y = np.random.randn(n_samples).astype(np.float32)

        # Split train/val
        n_train = 40
        return {
            "X_train": X[:n_train],
            "y_train": y[:n_train],
            "X_val": X[n_train:],
            "y_val": y[n_train:],
        }

    def test_train_transformer_basic(self, sample_data):
        """Test basic training pipeline."""
        result = train_transformer(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            epochs=2,
            batch_size=8,
            save_model=False,
        )

        assert "model" in result
        assert "history" in result
        assert "metrics" in result
        assert len(result["history"]["train_loss"]) == 2

    def test_train_with_validation(self, sample_data):
        """Test training with validation data."""
        result = train_transformer(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            epochs=2,
            save_model=False,
        )

        assert len(result["history"]["val_loss"]) > 0


class TestPredictTransformer:
    """Tests for prediction functions."""

    def test_predict_output_shape(self):
        """Test prediction output shape."""
        model = TransformerModel(input_size=5, d_model=32, nhead=4)

        X = np.random.randn(10, 8, 5).astype(np.float32)
        predictions = predict_transformer(model, X)

        assert predictions.shape == (10,)

    def test_predict_single_sample(self):
        """Test prediction for single sample."""
        model = TransformerModel(input_size=5, d_model=32, nhead=4)

        X = np.random.randn(1, 8, 5).astype(np.float32)
        predictions = predict_transformer(model, X)

        assert predictions.shape == (1,)


class TestSaveLoadTransformer:
    """Tests for saving and loading models."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading a model."""
        model = TransformerModel(input_size=5, d_model=32, nhead=4)
        model_path = str(tmp_path / "test_transformer.pt")

        # Save
        save_transformer_model(model, path=model_path)

        # Load
        loaded_model = load_transformer_model(model_path)

        assert loaded_model.input_size == 5
        assert loaded_model.d_model == 32

    def test_predictions_consistent_after_load(self, tmp_path):
        """Test that loaded model gives same predictions."""
        model = TransformerModel(input_size=5, d_model=32, nhead=4)
        model.eval()

        X = np.random.randn(5, 8, 5).astype(np.float32)
        original_preds = predict_transformer(model, X)

        model_path = str(tmp_path / "test_model.pt")
        save_transformer_model(model, path=model_path)
        loaded_model = load_transformer_model(model_path)

        loaded_preds = predict_transformer(loaded_model, X)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)
