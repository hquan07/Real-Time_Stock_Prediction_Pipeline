"""
Technical Indicators Calculator
RSI, MACD, Bollinger Bands, Moving Averages
"""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_ma(df: pd.DataFrame, windows: list = [5, 20, 50]) -> pd.DataFrame:
    """Calculate Moving Averages."""
    for w in windows:
        df[f'ma{w}'] = df['close'].rolling(window=w).mean()
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD indicator."""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
    return df


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators."""
    df = df.copy()
    df = calculate_ma(df, [5, 20, 50])
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    
    # Price change percentage
    df['price_change_pct'] = df['close'].pct_change() * 100
    
    # Volume ratio (vs 20-day average)
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    return df
