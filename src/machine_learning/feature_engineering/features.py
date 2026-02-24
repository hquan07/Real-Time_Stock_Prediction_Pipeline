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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_return_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)

    df = df.dropna()
    return df