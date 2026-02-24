"""
Pytest configuration and fixtures.
"""

import os
import sys
from datetime import date, datetime
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_price_data() -> pd.DataFrame:
    """Generate sample price data for testing."""
    np.random.seed(42)
    days = 100

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, days)
    prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame({
        "ticker": "AAPL",
        "date": dates,
        "open": prices * np.random.uniform(0.99, 1.01, days),
        "high": prices * np.random.uniform(1.01, 1.03, days),
        "low": prices * np.random.uniform(0.97, 0.99, days),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, days),
    })


@pytest.fixture(scope="session")
def sample_ohlcv() -> dict:
    """Sample OHLCV data dict."""
    return {
        "open": 150.0,
        "high": 155.0,
        "low": 148.0,
        "close": 152.0,
        "volume": 5000000,
    }


@pytest.fixture(scope="session")
def sample_kafka_message() -> dict:
    """Sample Kafka message."""
    return {
        "ticker": "AAPL",
        "timestamp": int(datetime.now().timestamp() * 1000),
        "price": {
            "open": 150.0,
            "high": 155.0,
            "low": 148.0,
            "close": 152.0,
            "volume": 5000000,
        },
        "dividends": 0.22,
        "splits": None,
        "earnings": {
            "eps_actual": 1.52,
            "eps_estimate": 1.48,
            "quarter": "Q4 2023",
        },
        "financials": {
            "total_revenue": 100000000000,
            "net_income": 25000000000,
            "operating_income": 30000000000,
        },
    }


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_db_session():
    """Mock database session."""
    with patch("src.database.db_connection.get_session") as mock:
        session = MagicMock()
        mock.return_value.__enter__ = MagicMock(return_value=session)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield session


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer."""
    with patch("src.data_ingestion.kafka_producer.producer.create_kafka_producer") as mock:
        producer = MagicMock()
        mock.return_value = producer
        yield producer


@pytest.fixture
def mock_yfinance():
    """Mock yfinance Ticker."""
    with patch("yfinance.Ticker") as mock:
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame({
            "Open": [150.0],
            "High": [155.0],
            "Low": [148.0],
            "Close": [152.0],
            "Volume": [5000000],
        })
        mock.return_value = ticker
        yield ticker


# ---------------------------------------------------------------------------
# Spark fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def spark_session():
    """Create Spark session for testing."""
    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder
            .master("local[1]")
            .appName("test")
            .config("spark.driver.memory", "512m")
            .getOrCreate()
        )
        yield spark
        spark.stop()
    except ImportError:
        pytest.skip("PySpark not installed")


# ---------------------------------------------------------------------------
# Cleanup fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def cleanup_model_cache():
    """Clear model cache after each test."""
    yield
    try:
        from src.machine_learning.inference.inference import ModelLoader
        ModelLoader.clear_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_db: marks tests that require database")
    config.addinivalue_line("markers", "requires_kafka: marks tests that require Kafka")
    config.addinivalue_line("markers", "requires_spark: marks tests that require Spark")
