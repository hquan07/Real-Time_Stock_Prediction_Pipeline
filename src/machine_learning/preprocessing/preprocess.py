"""
Data preprocessing module for Machine Learning pipeline.
Handles data cleaning, normalization, and train/test splitting for time series.
"""

import os
import pickle
from typing import Tuple, List, Optional, Dict, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "src/machine_learning/artifacts")


# ---------------------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------------------
def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw price data.
    
    Operations:
        - Remove duplicates
        - Sort by date
        - Handle missing values
        - Remove outliers
    """
    df = df.copy()

    # Remove duplicates
    if "date" in df.columns and "ticker" in df.columns:
        df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    elif "date" in df.columns:
        df = df.drop_duplicates(subset=["date"], keep="last")

    # Sort by date
    if "date" in df.columns:
        df = df.sort_values("date")

    # Handle missing values in price columns
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        if col in df.columns:
            # Forward fill then backward fill
            df[col] = df[col].ffill().bfill()

    # Handle missing volume
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)

    # Remove rows with all NaN prices
    price_cols_present = [c for c in price_cols if c in df.columns]
    if price_cols_present:
        df = df.dropna(subset=price_cols_present, how="all")

    logger.info(f"✅ Cleaned data: {len(df)} rows remaining")
    return df


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers from DataFrame.
    
    Methods:
        - iqr: Interquartile range method
        - zscore: Z-score method
    """
    df = df.copy()

    if column not in df.columns:
        return df

    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        df = df[(df[column] >= lower) & (df[column] <= upper)]

    elif method == "zscore":
        mean = df[column].mean()
        std = df[column].std()
        df = df[np.abs((df[column] - mean) / std) <= threshold]

    return df


# ---------------------------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------------------------
class DataScaler:
    """Wrapper for scikit-learn scalers with save/load functionality."""

    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize scaler.
        
        Args:
            scaler_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        """
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        self.scaler_type = scaler_type
        self.fitted = False
        self.feature_names = None

    def fit(self, df: pd.DataFrame, columns: List[str] = None) -> "DataScaler":
        """Fit scaler to data."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_names = columns
        self.scaler.fit(df[columns])
        self.fitted = True

        logger.info(f"✅ Scaler fitted on {len(columns)} columns")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        df = df.copy()
        df[self.feature_names] = self.scaler.transform(df[self.feature_names])
        return df

    def fit_transform(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, columns)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform data back to original scale."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        df = df.copy()
        df[self.feature_names] = self.scaler.inverse_transform(df[self.feature_names])
        return df

    def save(self, path: str = None) -> str:
        """Save scaler to file."""
        if path is None:
            os.makedirs(ARTIFACT_DIR, exist_ok=True)
            path = os.path.join(ARTIFACT_DIR, "scaler.pkl")

        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "scaler_type": self.scaler_type,
                "feature_names": self.feature_names,
                "fitted": self.fitted,
            }, f)

        logger.info(f"✅ Scaler saved to {path}")
        return path

    @classmethod
    def load(cls, path: str = None) -> "DataScaler":
        """Load scaler from file."""
        if path is None:
            path = os.path.join(ARTIFACT_DIR, "scaler.pkl")

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(data["scaler_type"])
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.fitted = data["fitted"]

        logger.info(f"✅ Scaler loaded from {path}")
        return instance


# ---------------------------------------------------------------------------
# Train/Test Split for Time Series
# ---------------------------------------------------------------------------
def train_test_split_ts(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    date_column: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time series aware train/test split.
    
    Splits data chronologically (no shuffling) to prevent data leakage.
    
    Args:
        df: DataFrame with time series data
        train_ratio: Proportion of data for training
        date_column: Name of the date column
    
    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values(date_column)

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(
        f"✅ Train/Test split: {len(train_df)} train, {len(test_df)} test "
        f"({train_ratio:.0%}/{1-train_ratio:.0%})"
    )

    return train_df, test_df


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    target_column_idx: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM/time series models.
    
    Args:
        data: 2D numpy array of features
        sequence_length: Number of time steps in each sequence
        target_column_idx: Index of target column (-1 for last column)
    
    Returns:
        Tuple of (X, y) where X has shape (samples, sequence_length, features)
    """
    X, y = [], []

    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length : i])
        y.append(data[i, target_column_idx])

    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------------------------------
def preprocess_for_training(
    df: pd.DataFrame,
    target_column: str = "close",
    feature_columns: List[str] = None,
    train_ratio: float = 0.8,
    scale: bool = True,
    save_scaler: bool = True,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for training.
    
    Args:
        df: Raw price DataFrame
        target_column: Column to predict
        feature_columns: Features to use (None = auto-detect)
        train_ratio: Train/test split ratio
        scale: Whether to scale features
        save_scaler: Whether to save scaler to disk
    
    Returns:
        Dict with keys: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Clean data
    df = clean_price_data(df)

    # Auto-detect feature columns
    if feature_columns is None:
        exclude = ["date", "ticker", "id"]
        feature_columns = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in ["float64", "int64"]
        ]

    logger.info(f"Using features: {feature_columns}")

    # Ensure target is in features
    if target_column not in feature_columns:
        feature_columns.append(target_column)

    # Select columns
    df_features = df[feature_columns].copy()

    # Handle any remaining NaN
    df_features = df_features.dropna()

    # Split
    train_df, test_df = train_test_split_ts(df, train_ratio)

    # Scale
    scaler = None
    if scale:
        scaler = DataScaler("standard")
        train_df = scaler.fit_transform(train_df[feature_columns])
        test_df = scaler.transform(test_df[feature_columns])

        if save_scaler:
            scaler.save()

    else:
        train_df = train_df[feature_columns]
        test_df = test_df[feature_columns]

    # Separate X and y
    feature_cols = [c for c in feature_columns if c != target_column]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_column].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_column].values

    logger.info(f"✅ Preprocessing complete: X_train shape {X_train.shape}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols,
        "target_column": target_column,
    }


def preprocess_for_inference(
    df: pd.DataFrame,
    scaler_path: str = None,
    feature_columns: List[str] = None,
) -> np.ndarray:
    """
    Preprocess data for model inference.
    
    Args:
        df: Raw data DataFrame
        scaler_path: Path to saved scaler
        feature_columns: Feature columns to use
    
    Returns:
        Preprocessed numpy array ready for model input.
    """
    # Clean data
    df = clean_price_data(df)

    # Load scaler
    if scaler_path:
        scaler = DataScaler.load(scaler_path)
        if feature_columns is None:
            feature_columns = scaler.feature_names
        df = scaler.transform(df)

    # Select features
    if feature_columns:
        df = df[feature_columns]

    return df.values
