"""Leakage-safe feature engineering for per-symbol price series."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


SYMBOL_COLUMN = "ticker"
TIME_COLUMN = "date"
TARGET_COLUMN = "target_return"


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    required = {SYMBOL_COLUMN, TIME_COLUMN, "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    result = df.copy()
    result[SYMBOL_COLUMN] = result[SYMBOL_COLUMN].astype(str).str.upper()
    result[TIME_COLUMN] = pd.to_datetime(result[TIME_COLUMN], utc=True)
    return result.sort_values([SYMBOL_COLUMN, TIME_COLUMN]).reset_index(drop=True)


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    result = _ordered(df)
    previous_close = result.groupby(SYMBOL_COLUMN, sort=False)["close"].shift(1)
    result["log_return"] = np.log(result["close"] / previous_close)
    return result


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    result = _ordered(df)
    grouped = result.groupby(SYMBOL_COLUMN, sort=False)
    result["ma_5"] = grouped["close"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    result["ma_10"] = grouped["close"].transform(
        lambda values: values.rolling(10, min_periods=10).mean()
    )
    result["volatility_10"] = grouped["log_return"].transform(
        lambda values: values.rolling(10, min_periods=10).std()
    )
    return result


def add_lag_features(
    df: pd.DataFrame, lags: Iterable[int] = (1, 5, 10)
) -> pd.DataFrame:
    result = _ordered(df)
    grouped_close = result.groupby(SYMBOL_COLUMN, sort=False)["close"]
    for lag in lags:
        if lag <= 0:
            raise ValueError("Lags must be positive integers")
        result[f"close_lag_{lag}"] = grouped_close.shift(lag)
    return result


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    result = _ordered(df)
    result[TARGET_COLUMN] = result.groupby(SYMBOL_COLUMN, sort=False)[
        "log_return"
    ].shift(-1)
    return result


def build_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Build features without allowing rolling, lag or target values across symbols."""
    result = add_return_features(df)
    result = add_rolling_features(result)
    result = add_lag_features(result)
    result = add_target(result)

    feature_columns = [
        "log_return",
        "ma_5",
        "ma_10",
        "volatility_10",
        "close_lag_1",
        "close_lag_5",
        "close_lag_10",
    ]
    required = feature_columns + ([TARGET_COLUMN] if is_training else [])
    return result.dropna(subset=required).reset_index(drop=True)
