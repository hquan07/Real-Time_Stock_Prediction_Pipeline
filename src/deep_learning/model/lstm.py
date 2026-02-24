"""
LSTM model for stock price prediction.
Uses PyTorch for building and training LSTM neural networks.
"""

import os
import json
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch not installed. LSTM model will not be available.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DL_MODEL_DIR = os.getenv("DL_MODEL_DIR", "src/deep_learning/artifacts")
DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset for Time Series
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class StockDataset(Dataset):
        """PyTorch Dataset for stock price sequences."""

        def __init__(self, X: np.ndarray, y: np.ndarray):
            """
            Initialize dataset.
            
            Args:
                X: Feature sequences of shape (samples, sequence_length, features)
                y: Target values of shape (samples,)
            """
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """
        LSTM network for stock price prediction.
        
        Architecture:
            - LSTM layers with dropout
            - Fully connected output layer
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            output_size: int = 1,
        ):
            """
            Initialize LSTM model.
            
            Args:
                input_size: Number of input features
                hidden_size: Number of hidden units per LSTM layer
                num_layers: Number of LSTM layers
                dropout: Dropout rate for regularization
                output_size: Number of output values (1 for regression)
            """
            super(LSTMModel, self).__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

            # Fully connected output
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor of shape (batch, sequence_length, features)
            
            Returns:
                Output tensor of shape (batch, output_size)
            """
            # LSTM forward
            lstm_out, (h_n, c_n) = self.lstm(x)

            # Use last hidden state
            out = self.fc(h_n[-1])

            return out.squeeze(-1)
else:
    # Placeholder classes when PyTorch not available
    class StockDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTMModel. Install with: pip install torch")

    class LSTMModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTMModel. Install with: pip install torch")


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------
def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    save_model: bool = True,
) -> Dict[str, Any]:
    """
    Train LSTM model on stock data.
    
    Args:
        X_train: Training features (samples, sequence_length, features)
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        early_stopping_patience: Epochs to wait for improvement
        save_model: Whether to save best model
    
    Returns:
        Dict with model, history, and metrics.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = StockDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training history
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    logger.info(f"🚀 Starting LSTM training on {DEVICE} for {epochs} epochs")

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        if val_loader:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)

                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            history["val_loss"].append(avg_val_loss)

            scheduler.step(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"⏹ Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )
        else:
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}")

    # Load best model if early stopping was used
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Calculate final metrics
    final_metrics = {
        "final_train_loss": history["train_loss"][-1],
        "best_val_loss": best_val_loss if val_loader else None,
        "epochs_trained": len(history["train_loss"]),
        "device": DEVICE,
    }

    # Save model
    if save_model:
        save_path = save_lstm_model(model, final_metrics)
        final_metrics["model_path"] = save_path

    logger.info(f"✅ LSTM training complete. Best val loss: {best_val_loss:.6f}")

    return {
        "model": model,
        "history": history,
        "metrics": final_metrics,
    }


# ---------------------------------------------------------------------------
# Model Saving/Loading
# ---------------------------------------------------------------------------
def save_lstm_model(
    model: "LSTMModel",
    metrics: Dict = None,
    path: str = None,
) -> str:
    """Save LSTM model to disk."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    if path is None:
        os.makedirs(DL_MODEL_DIR, exist_ok=True)
        path = os.path.join(DL_MODEL_DIR, "lstm_model.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "input_size": model.lstm.input_size,
    }, path)

    # Save metrics
    if metrics:
        metrics_path = path.replace(".pt", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    logger.info(f"✅ LSTM model saved to {path}")
    return path


def load_lstm_model(path: str = None) -> "LSTMModel":
    """Load LSTM model from disk."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    if path is None:
        path = os.path.join(DL_MODEL_DIR, "lstm_model.pt")

    checkpoint = torch.load(path, map_location=DEVICE)

    model = LSTMModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    logger.info(f"✅ LSTM model loaded from {path}")
    return model


# ---------------------------------------------------------------------------
# Prediction Functions
# ---------------------------------------------------------------------------
def predict_lstm(
    model: "LSTMModel",
    X: np.ndarray,
) -> np.ndarray:
    """
    Make predictions with LSTM model.
    
    Args:
        model: Trained LSTM model
        X: Input features (samples, sequence_length, features)
    
    Returns:
        Predictions array.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    model.eval()
    X_tensor = torch.FloatTensor(X).to(DEVICE)

    with torch.no_grad():
        predictions = model(X_tensor)

    return predictions.cpu().numpy()


def predict_next_price_lstm(
    sequence: np.ndarray,
    model_path: str = None,
) -> float:
    """
    Predict next price using saved LSTM model.
    
    Args:
        sequence: Input sequence (sequence_length, features)
        model_path: Path to saved model
    
    Returns:
        Predicted price.
    """
    model = load_lstm_model(model_path)

    # Add batch dimension
    X = sequence.reshape(1, *sequence.shape)

    prediction = predict_lstm(model, X)
    return float(prediction[0])
