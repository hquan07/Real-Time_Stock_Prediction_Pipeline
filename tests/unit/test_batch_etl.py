"""
Unit tests for Batch ETL pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch

# Skip tests if PySpark not available
pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType, DateType
)
import datetime


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("TestBatchETL")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def sample_stock_data(spark):
    """Create sample stock data DataFrame."""
    schema = StructType([
        StructField("ticker", StringType(), False),
        StructField("date", DateType(), False),
        StructField("open", FloatType(), True),
        StructField("high", FloatType(), True),
        StructField("low", FloatType(), True),
        StructField("close", FloatType(), True),
        StructField("volume", IntegerType(), True),
    ])

    data = [
        ("AAPL", datetime.date(2023, 1, 1), 150.0, 155.0, 148.0, 153.0, 1000000),
        ("AAPL", datetime.date(2023, 1, 2), 153.0, 157.0, 151.0, 155.0, 1200000),
        ("AAPL", datetime.date(2023, 1, 3), 155.0, 158.0, 154.0, 156.0, 1100000),
        ("AAPL", datetime.date(2023, 1, 4), 156.0, 160.0, 155.0, 159.0, 1300000),
        ("AAPL", datetime.date(2023, 1, 5), 159.0, 161.0, 157.0, 160.0, 1400000),
        ("MSFT", datetime.date(2023, 1, 1), 250.0, 255.0, 248.0, 253.0, 800000),
        ("MSFT", datetime.date(2023, 1, 2), 253.0, 257.0, 251.0, 255.0, 850000),
        ("MSFT", datetime.date(2023, 1, 3), 255.0, 258.0, 254.0, 256.0, 900000),
        ("MSFT", datetime.date(2023, 1, 4), 256.0, 260.0, 255.0, 259.0, 950000),
        ("MSFT", datetime.date(2023, 1, 5), 259.0, 262.0, 257.0, 261.0, 1000000),
    ]

    return spark.createDataFrame(data, schema)


class TestComputeSMA:
    """Tests for SMA computation."""

    def test_sma_calculation(self, sample_stock_data):
        """Test SMA is calculated correctly."""
        from src.streaming.batch.batch_etl import compute_sma

        result = compute_sma(sample_stock_data, "close", 3)
        
        assert "sma_3" in result.columns
        assert result.filter("sma_3 IS NOT NULL").count() > 0

    def test_sma_column_name(self, sample_stock_data):
        """Test SMA column is named correctly."""
        from src.streaming.batch.batch_etl import compute_sma

        result = compute_sma(sample_stock_data, "close", 5)
        
        assert "sma_5" in result.columns


class TestComputeEMA:
    """Tests for EMA computation."""

    def test_ema_calculation(self, sample_stock_data):
        """Test EMA is calculated."""
        from src.streaming.batch.batch_etl import compute_ema

        result = compute_ema(sample_stock_data, "close", 3)
        
        assert "ema_3" in result.columns
        assert result.filter("ema_3 IS NOT NULL").count() > 0


class TestComputeRSI:
    """Tests for RSI computation."""

    def test_rsi_calculation(self, sample_stock_data):
        """Test RSI is calculated."""
        from src.streaming.batch.batch_etl import compute_rsi

        result = compute_rsi(sample_stock_data, "close", 3)
        
        assert "rsi" in result.columns

    def test_rsi_range(self, sample_stock_data):
        """Test RSI values are in valid range 0-100."""
        from src.streaming.batch.batch_etl import compute_rsi

        result = compute_rsi(sample_stock_data, "close", 3)
        
        # RSI should be between 0 and 100
        rsi_values = [row.rsi for row in result.filter("rsi IS NOT NULL").collect()]
        
        for val in rsi_values:
            assert 0 <= val <= 100


class TestComputeBollingerBands:
    """Tests for Bollinger Bands computation."""

    def test_bollinger_bands_columns(self, sample_stock_data):
        """Test Bollinger Bands adds correct columns."""
        from src.streaming.batch.batch_etl import compute_bollinger_bands

        result = compute_bollinger_bands(sample_stock_data, "close", 3, 2.0)
        
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns

    def test_bollinger_bands_ordering(self, sample_stock_data):
        """Test upper > middle > lower."""
        from src.streaming.batch.batch_etl import compute_bollinger_bands

        result = compute_bollinger_bands(sample_stock_data, "close", 3, 2.0)
        
        rows = result.filter(
            "bb_upper IS NOT NULL AND bb_middle IS NOT NULL AND bb_lower IS NOT NULL"
        ).collect()
        
        for row in rows:
            assert row.bb_upper >= row.bb_middle
            assert row.bb_middle >= row.bb_lower


class TestComputeDailyReturns:
    """Tests for daily returns computation."""

    def test_daily_returns_column(self, sample_stock_data):
        """Test daily returns column is added."""
        from src.streaming.batch.batch_etl import compute_daily_returns

        result = compute_daily_returns(sample_stock_data, "close")
        
        assert "daily_return" in result.columns

    def test_first_row_null_return(self, sample_stock_data):
        """Test first row has null return (no previous day)."""
        from src.streaming.batch.batch_etl import compute_daily_returns

        result = compute_daily_returns(sample_stock_data, "close")
        
        # Filter by ticker and get first date
        first_aapl = result.filter("ticker = 'AAPL'").orderBy("date").first()
        
        assert first_aapl.daily_return is None


class TestCreateSparkSession:
    """Tests for Spark session creation."""

    def test_create_session(self):
        """Test session creation."""
        from src.streaming.batch.batch_etl import create_spark_session

        # This would create a new session, so just test the function exists
        assert callable(create_spark_session)
