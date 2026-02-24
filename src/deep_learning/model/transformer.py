"""
Transformer model for stock price prediction.
Uses PyTorch for building and training Transformer architecture.
"""

import os
import json
import math
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
    logger.warning("⚠️ PyTorch not installed. Transformer model will not be available.")


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
# Positional Encoding
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        """
        Positional encoding for Transformer.
        Adds positional information to embeddings using sine and cosine functions.
        """

        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            """
            Initialize positional encoding.
            
            Args:
                d_model: Dimension of the model
                max_len: Maximum sequence length
                dropout: Dropout rate
            """
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)

            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Add positional encoding to input.
            
            Args:
                x: Input tensor of shape (batch, seq_len, d_model)
            
            Returns:
                Tensor with positional encoding added.
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class TransformerModel(nn.Module):
        """
        Transformer encoder for stock price prediction.
        
        Uses self-attention mechanism to capture long-range dependencies
        in time series data.
        
        Architecture:
            - Input projection layer
            - Positional encoding
            - Transformer encoder layers
            - Output projection layer
        """

        def __init__(
            self,
            input_size: int,
            d_model: int = 64,
            nhead: int = 4,
            num_encoder_layers: int = 3,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            output_size: int = 1,
        ):
            """
            Initialize Transformer model.
            
            Args:
                input_size: Number of input features
                d_model: Dimension of the model (embedding size)
                nhead: Number of attention heads
                num_encoder_layers: Number of encoder layers
                dim_feedforward: Dimension of feedforward network
                dropout: Dropout rate for regularization
                output_size: Number of output values (1 for regression)
            """
            super(TransformerModel, self).__init__()

            self.input_size = input_size
            self.d_model = d_model
            self.nhead = nhead
            self.num_encoder_layers = num_encoder_layers

            # Input projection: project input features to d_model dimensions
            self.input_projection = nn.Linear(input_size, d_model)

            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=num_encoder_layers
            )

            # Output projection
            self.fc_out = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor of shape (batch, sequence_length, features)
            
            Returns:
                Output tensor of shape (batch, output_size)
            """
            # Project input to d_model dimensions
            x = self.input_projection(x)  # (batch, seq_len, d_model)

            # Add positional encoding
            x = self.pos_encoder(x)

            # Transformer encoder
            x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

            # Use the last position's output for prediction
            x = x[:, -1, :]  # (batch, d_model)

            # Output projection
            out = self.fc_out(x)  # (batch, output_size)

            return out.squeeze(-1)

else:
    # Placeholder classes when PyTorch not available
    class StockDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Install with: pip install torch")

    class PositionalEncoding:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Install with: pip install torch")

    class TransformerModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required. Install with: pip install torch")


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------
def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 3,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    save_model: bool = True,
) -> Dict[str, Any]:
    """
    Train Transformer model on stock data.
    
    Args:
        X_train: Training features (samples, sequence_length, features)
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        d_model: Model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        dim_feedforward: Feedforward dimension
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
    model = TransformerModel(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Transformer model has {total_params:,} trainable parameters")

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

    logger.info(f"🚀 Starting Transformer training on {DEVICE} for {epochs} epochs")

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

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        "total_params": total_params,
        "device": DEVICE,
        "model_type": "transformer",
    }

    # Save model
    if save_model:
        save_path = save_transformer_model(model, final_metrics)
        final_metrics["model_path"] = save_path

    logger.info(f"✅ Transformer training complete. Best val loss: {best_val_loss:.6f}")

    return {
        "model": model,
        "history": history,
        "metrics": final_metrics,
    }


# ---------------------------------------------------------------------------
# Model Saving/Loading
# ---------------------------------------------------------------------------
def save_transformer_model(
    model: "TransformerModel",
    metrics: Dict = None,
    path: str = None,
) -> str:
    """Save Transformer model to disk."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    if path is None:
        os.makedirs(DL_MODEL_DIR, exist_ok=True)
        path = os.path.join(DL_MODEL_DIR, "transformer_model.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": model.input_size,
        "d_model": model.d_model,
        "nhead": model.nhead,
        "num_encoder_layers": model.num_encoder_layers,
    }, path)

    # Save metrics
    if metrics:
        metrics_path = path.replace(".pt", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    logger.info(f"✅ Transformer model saved to {path}")
    return path


def load_transformer_model(path: str = None) -> "TransformerModel":
    """Load Transformer model from disk."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    if path is None:
        path = os.path.join(DL_MODEL_DIR, "transformer_model.pt")

    checkpoint = torch.load(path, map_location=DEVICE)

    model = TransformerModel(
        input_size=checkpoint["input_size"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_encoder_layers=checkpoint["num_encoder_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    logger.info(f"✅ Transformer model loaded from {path}")
    return model


# ---------------------------------------------------------------------------
# Prediction Functions
# ---------------------------------------------------------------------------
def predict_transformer(
    model: "TransformerModel",
    X: np.ndarray,
) -> np.ndarray:
    """
    Make predictions with Transformer model.
    
    Args:
        model: Trained Transformer model
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


def predict_next_price_transformer(
    sequence: np.ndarray,
    model_path: str = None,
) -> float:
    """
    Predict next price using saved Transformer model.
    
    Args:
        sequence: Input sequence (sequence_length, features)
        model_path: Path to saved model
    
    Returns:
        Predicted price.
    """
    model = load_transformer_model(model_path)

    # Add batch dimension
    X = sequence.reshape(1, *sequence.shape)

    prediction = predict_transformer(model, X)
    return float(prediction[0])
