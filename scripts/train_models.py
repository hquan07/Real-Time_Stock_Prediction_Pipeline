#!/usr/bin/env python3
"""
Train ML and DL models with data from PostgreSQL.
- RandomForest: Predict next-day price using technical indicators
- LSTM: Predict 7-day price sequence
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. LSTM training will be skipped.")

# DB Config
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "stockdb",
    "user": "postgres",
    "password": "Huyquan1607"
}

ARTIFACTS_DIR = "src/machine_learning/artifacts"
DL_ARTIFACTS_DIR = "src/deep_learning/artifacts"


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for ML features."""
    df = df.copy()
    
    # Moving Averages
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # Price ratios
    df['price_ma5_ratio'] = df['close'] / df['ma5']
    df['price_ma20_ratio'] = df['close'] / df['ma20']
    
    # Volatility (20-day rolling std)
    df['volatility'] = df['close'].rolling(window=20).std()
    
    # RSI (14-day)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Price change
    df['price_change'] = df['close'].pct_change()
    
    # Target: next day close price
    df['target'] = df['close'].shift(-1)
    
    return df


def load_price_data(ticker: str = None) -> pd.DataFrame:
    """Load price data from PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    
    if ticker:
        query = f"SELECT * FROM price_history WHERE ticker = '{ticker}' ORDER BY date"
    else:
        query = "SELECT * FROM price_history ORDER BY ticker, date"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} price records")
    return df


def train_random_forest(ticker: str = "AAPL") -> dict:
    """Train RandomForest model for price prediction."""
    logger.info(f"🌲 Training RandomForest for {ticker}...")
    
    # Load and prepare data
    df = load_price_data(ticker)
    df = calculate_technical_indicators(df)
    
    # Features
    feature_cols = ['ma5', 'ma20', 'price_ma5_ratio', 'price_ma20_ratio', 
                    'volatility', 'rsi', 'macd', 'macd_signal', 
                    'volume_ratio', 'price_change', 'close']
    
    # Drop NaN rows
    df = df.dropna(subset=feature_cols + ['target'])
    
    X = df[feature_cols].values
    y = df['target'].values
    
    # Train/test split (keep time order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
    
    logger.info(f"  Train RMSE: ${train_rmse:.2f}")
    logger.info(f"  Test RMSE: ${test_rmse:.2f}")
    logger.info(f"  Test MAE: ${test_mae:.2f}")
    logger.info(f"  Test MAPE: {mape:.2f}%")
    
    # Save model
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_DIR, f"rf_model_{ticker}.pkl")
    
    model_data = {
        "model": model,
        "feature_cols": feature_cols,
        "ticker": ticker,
        "metrics": {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "mape": mape
        },
        "trained_at": datetime.now().isoformat()
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    logger.info(f"  ✅ Model saved: {model_path}")
    
    return model_data


# LSTM Model
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=7):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, output_size)
            )
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out

    class StockDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


def create_sequences(data, seq_length=30, pred_length=7):
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_length, 0])  # Close prices
    return np.array(X), np.array(y)


def train_lstm(ticker: str = "AAPL") -> dict:
    """Train LSTM model for 7-day price prediction."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available. Skipping LSTM training.")
        return None
    
    logger.info(f"🧠 Training LSTM for {ticker}...")
    
    # Load data
    df = load_price_data(ticker)
    
    # Features: close, volume (normalized)
    features = df[['close', 'volume']].values
    
    # Normalize
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    seq_length = 30
    pred_length = 7
    X, y = create_sequences(features_scaled, seq_length, pred_length)
    
    # Train/val split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create dataloaders
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel(input_size=2, hidden_size=64, num_layers=2, output_size=pred_length)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training
    epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.6f}, Val Loss={val_loss:.6f}")
    
    # Save model
    os.makedirs(DL_ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(DL_ARTIFACTS_DIR, f"lstm_model_{ticker}.pt")
    
    # Convert val_loss to RMSE in original scale
    close_scaler = MinMaxScaler()
    close_scaler.fit(df[['close']].values)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_params": {
            "min": scaler.data_min_.tolist(),
            "max": scaler.data_max_.tolist()
        },
        "close_scaler_params": {
            "min": close_scaler.data_min_.tolist(),
            "max": close_scaler.data_max_.tolist()
        },
        "config": {
            "input_size": 2,
            "hidden_size": 64,
            "num_layers": 2,
            "seq_length": seq_length,
            "pred_length": pred_length
        },
        "ticker": ticker,
        "val_loss": best_val_loss,
        "trained_at": datetime.now().isoformat()
    }, model_path)
    
    logger.info(f"  ✅ LSTM saved: {model_path}")
    
    return {"model_path": model_path, "val_loss": best_val_loss}


def main():
    """Train all models."""
    logger.info("=" * 60)
    logger.info("🚀 Training ML/DL Models")
    logger.info("=" * 60)
    
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "WMT"]
    
    results = {"random_forest": {}, "lstm": {}}
    
    for ticker in tickers:
        logger.info(f"\n📊 Training models for {ticker}...")
        
        # Train RandomForest
        try:
            rf_result = train_random_forest(ticker)
            results["random_forest"][ticker] = rf_result["metrics"]
        except Exception as e:
            logger.error(f"  ❌ RandomForest failed for {ticker}: {e}")
        
        # Train LSTM
        try:
            lstm_result = train_lstm(ticker)
            if lstm_result:
                results["lstm"][ticker] = lstm_result
        except Exception as e:
            logger.error(f"  ❌ LSTM failed for {ticker}: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training Complete!")
    logger.info("=" * 60)
    
    # Summary
    logger.info("\n📊 RandomForest Results:")
    for ticker, metrics in results["random_forest"].items():
        logger.info(f"  {ticker}: RMSE=${metrics['test_rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
    
    if results["lstm"]:
        logger.info("\n🧠 LSTM Results:")
        for ticker, data in results["lstm"].items():
            logger.info(f"  {ticker}: Val Loss={data['val_loss']:.6f}")


if __name__ == "__main__":
    main()