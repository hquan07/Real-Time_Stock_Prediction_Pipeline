"""
Unit tests for feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd


class TestReturnFeatures:
    """Tests for return features."""

    def test_add_return_features(self, sample_price_data):
        """Test log return calculation."""
        from src.machine_learning.feature_engineering.features import add_return_features

        result = add_return_features(sample_price_data)

        assert "log_return" in result.columns
        # First value should be NaN
        assert pd.isna(result["log_return"].iloc[0])
        # Others should be valid
        assert not pd.isna(result["log_return"].iloc[1])


class TestRollingFeatures:
    """Tests for rolling features."""

    def test_add_rolling_features(self, sample_price_data):
        """Test rolling feature calculation."""
        from src.machine_learning.feature_engineering.features import (
            add_return_features,
            add_rolling_features,
        )

        # Need returns first
        df = add_return_features(sample_price_data)
        result = add_rolling_features(df)

        assert "ma_5" in result.columns
        assert "ma_10" in result.columns
        assert "volatility_10" in result.columns

    def test_moving_average_values(self, sample_price_data):
        """Test that moving averages are computed correctly."""
        from src.machine_learning.feature_engineering.features import (
            add_return_features,
            add_rolling_features,
        )

        df = add_return_features(sample_price_data)
        result = add_rolling_features(df)

        # MA5 should equal mean of last 5 closes
        for i in range(5, len(result)):
            expected = result["close"].iloc[i - 4 : i + 1].mean()
            actual = result["ma_5"].iloc[i]
            np.testing.assert_almost_equal(actual, expected, decimal=5)


class TestLagFeatures:
    """Tests for lag features."""

    def test_add_lag_features(self, sample_price_data):
        """Test lag feature creation."""
        from src.machine_learning.feature_engineering.features import add_lag_features

        result = add_lag_features(sample_price_data, lags=[1, 5])

        assert "close_lag_1" in result.columns
        assert "close_lag_5" in result.columns

    def test_lag_values_correct(self, sample_price_data):
        """Test that lag values are correct."""
        from src.machine_learning.feature_engineering.features import add_lag_features

        result = add_lag_features(sample_price_data, lags=[1])

        # lag_1 should be previous close
        for i in range(1, len(result)):
            expected = sample_price_data["close"].iloc[i - 1]
            actual = result["close_lag_1"].iloc[i]
            assert actual == expected


class TestBuildFeatures:
    """Tests for full feature pipeline."""

    def test_build_features_complete(self, sample_price_data):
        """Test complete feature building."""
        from src.machine_learning.feature_engineering.features import build_features

        result = build_features(sample_price_data)

        # Should have all features
        expected_cols = [
            "log_return",
            "ma_5",
            "ma_10",
            "volatility_10",
            "close_lag_1",
            "close_lag_5",
            "close_lag_10",
        ]

        for col in expected_cols:
            assert col in result.columns

        # No NaN values (dropped)
        assert not result.isna().any().any()

    def test_build_features_length(self, sample_price_data):
        """Test that feature building reduces rows appropriately."""
        from src.machine_learning.feature_engineering.features import build_features

        result = build_features(sample_price_data)

        # Should have fewer rows due to rolling/lag NaN removal
        assert len(result) < len(sample_price_data)
        assert len(result) > 0

    def test_features_and_target_do_not_cross_tickers(self, sample_price_data):
        from src.machine_learning.feature_engineering.features import build_features

        aapl = sample_price_data.copy()
        msft = sample_price_data.copy()
        msft["ticker"] = "MSFT"
        msft["close"] = msft["close"] * 10
        mixed = pd.concat([aapl, msft]).sample(frac=1, random_state=7)

        result = build_features(mixed, is_training=True)

        for ticker, group in result.groupby("ticker"):
            source = aapl if ticker == "AAPL" else msft
            expected_first_lag = source.sort_values("date")["close"].iloc[9]
            assert group.iloc[0]["close_lag_1"] == expected_first_lag
            assert group.iloc[-1]["target_return"] == pytest.approx(
                np.log(
                    source.sort_values("date")["close"].iloc[-1]
                    / source.sort_values("date")["close"].iloc[-2]
                )
            )
