import pandas as pd
import numpy as np

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date")
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["volatility_10"] = df["log_return"].rolling(10).std()
    return df


def add_lag_features(df: pd.DataFrame, lags=[1, 5, 10]) -> pd.DataFrame:
    for lag in lags:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    # We predict the next day's log return
    df["target_return"] = df["log_return"].shift(-1)
    return df

def build_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    df = add_return_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = add_target(df)

    if is_training:
        df = df.dropna()
    else:
        # For real-time, target_return will be NaN for the latest row. 
        # We only drop rows that have NaN in features (e.g. the first 10 rows due to lag_10).
        # We can drop subset of feature columns.
        feature_cols = [c for c in df.columns if c not in ['target_return', 'date', 'ticker', 'event_time']]
        df = df.dropna(subset=feature_cols)
        
    return df