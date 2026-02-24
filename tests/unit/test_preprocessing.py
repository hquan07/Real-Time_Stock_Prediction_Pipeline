"""
Unit tests for ML preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd


class TestCleanPriceData:
    """Tests for price data cleaning."""

    def test_clean_removes_duplicates(self, sample_price_data):
        """Test that duplicates are removed."""
        from src.machine_learning.preprocessing.preprocess import clean_price_data

        # Add duplicate
        df = pd.concat([sample_price_data, sample_price_data.iloc[[0]]])

        cleaned = clean_price_data(df)

        assert len(cleaned) == len(sample_price_data)

    def test_clean_sorts_by_date(self, sample_price_data):
        """Test that data is sorted by date."""
        from src.machine_learning.preprocessing.preprocess import clean_price_data

        # Shuffle
        df = sample_price_data.sample(frac=1)

        cleaned = clean_price_data(df)

        dates = cleaned["date"].tolist()
        assert dates == sorted(dates)

    def test_clean_handles_nan(self, sample_price_data):
        """Test that NaN values are handled."""
        from src.machine_learning.preprocessing.preprocess import clean_price_data

        df = sample_price_data.copy()
        df.loc[5, "close"] = np.nan

        cleaned = clean_price_data(df)

        # NaN should be filled
        assert not cleaned["close"].isna().any()


class TestDataScaler:
    """Tests for DataScaler class."""

    def test_scaler_fit_transform(self, sample_price_data):
        """Test scaler fit and transform."""
        from src.machine_learning.preprocessing.preprocess import DataScaler

        scaler = DataScaler("standard")
        columns = ["open", "high", "low", "close"]

        result = scaler.fit_transform(sample_price_data, columns)

        # Check that values are scaled (mean ~0, std ~1)
        for col in columns:
            assert abs(result[col].mean()) < 0.1
            assert abs(result[col].std() - 1.0) < 0.1

    def test_scaler_inverse_transform(self, sample_price_data):
        """Test scaler inverse transform."""
        from src.machine_learning.preprocessing.preprocess import DataScaler

        scaler = DataScaler("standard")
        columns = ["close"]

        original = sample_price_data.copy()
        scaled = scaler.fit_transform(sample_price_data[columns], columns)
        unscaled = scaler.inverse_transform(scaled)

        np.testing.assert_array_almost_equal(
            original[columns].values,
            unscaled[columns].values,
            decimal=5,
        )

    def test_scaler_minmax(self, sample_price_data):
        """Test MinMax scaler."""
        from src.machine_learning.preprocessing.preprocess import DataScaler

        scaler = DataScaler("minmax")
        columns = ["close"]

        result = scaler.fit_transform(sample_price_data, columns)

        # Check values are in [0, 1]
        assert result["close"].min() >= 0
        assert result["close"].max() <= 1


class TestTrainTestSplit:
    """Tests for time series train/test split."""

    def test_split_ratio(self, sample_price_data):
        """Test that split ratio is respected."""
        from src.machine_learning.preprocessing.preprocess import train_test_split_ts

        train, test = train_test_split_ts(sample_price_data, train_ratio=0.8)

        expected_train_size = int(len(sample_price_data) * 0.8)

        assert len(train) == expected_train_size
        assert len(test) == len(sample_price_data) - expected_train_size

    def test_split_chronological(self, sample_price_data):
        """Test that split is chronological."""
        from src.machine_learning.preprocessing.preprocess import train_test_split_ts

        train, test = train_test_split_ts(sample_price_data, train_ratio=0.8)

        # Last train date should be before first test date
        assert train["date"].max() < test["date"].min()


class TestCreateSequences:
    """Tests for sequence creation."""

    def test_sequence_shape(self):
        """Test output sequence shape."""
        from src.machine_learning.preprocessing.preprocess import create_sequences

        data = np.random.rand(100, 5)
        seq_length = 10

        X, y = create_sequences(data, seq_length)

        assert X.shape == (90, 10, 5)
        assert y.shape == (90,)

    def test_sequence_values(self):
        """Test sequence values are correct."""
        from src.machine_learning.preprocessing.preprocess import create_sequences

        data = np.arange(20).reshape(20, 1)
        seq_length = 5

        X, y = create_sequences(data, seq_length, target_column_idx=0)

        # First sequence should be [0,1,2,3,4], target should be 5
        np.testing.assert_array_equal(X[0].flatten(), [0, 1, 2, 3, 4])
        assert y[0] == 5
