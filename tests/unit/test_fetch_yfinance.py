"""
Unit tests for yfinance data fetching.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


class TestFetchPrice:
    """Tests for price fetching."""

    def test_fetch_price_success(self, mock_yfinance):
        """Test successful price fetch."""
        from src.data_ingestion.fetcher.price import fetch_price

        result = fetch_price("AAPL")

        assert result is not None
        assert "open" in result
        assert "high" in result
        assert "low" in result
        assert "close" in result
        assert "volume" in result

    @patch("yfinance.Ticker")
    def test_fetch_price_empty_history(self, mock_ticker):
        """Test handling empty history."""
        from src.data_ingestion.fetcher.price import fetch_price

        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame()
        mock_ticker.return_value = ticker

        result = fetch_price("INVALID")

        assert result is None

    @patch("yfinance.Ticker")
    def test_fetch_price_exception(self, mock_ticker):
        """Test handling exceptions."""
        from src.data_ingestion.fetcher.price import fetch_price

        mock_ticker.side_effect = Exception("API Error")

        result = fetch_price("AAPL")

        assert result is None


class TestFetchDividends:
    """Tests for dividend fetching."""

    @patch("yfinance.Ticker")
    def test_fetch_dividends_success(self, mock_ticker):
        """Test successful dividend fetch."""
        from src.data_ingestion.fetcher.dividends import fetch_dividends

        ticker = MagicMock()
        ticker.dividends = pd.Series([0.22, 0.23], index=pd.date_range("2023-01-01", periods=2))
        mock_ticker.return_value = ticker

        result = fetch_dividends("AAPL")

        assert result is not None
        assert isinstance(result, float)


class TestTickerList:
    """Tests for ticker list fetching."""

    @patch("pandas.read_csv")
    def test_get_ticker_list_success(self, mock_read_csv):
        """Test successful ticker list fetch."""
        # Mock CSV responses
        mock_read_csv.return_value = pd.DataFrame({
            "Symbol": ["AAPL", "MSFT", "GOOGL"]
        })

        from src.data_ingestion.fetcher.tickers import get_ticker_list

        result = get_ticker_list()

        assert isinstance(result, list)
        assert len(result) > 0


class TestBuildStockMessage:
    """Tests for building stock messages."""

    @patch("src.data_ingestion.fetcher.price.fetch_price")
    @patch("src.data_ingestion.fetcher.dividends.fetch_dividends")
    @patch("src.data_ingestion.fetcher.earnings.fetch_earnings")
    @patch("src.data_ingestion.fetcher.splits.fetch_splits")
    @patch("src.data_ingestion.fetcher.financials.fetch_financials")
    def test_build_stock_message_structure(
        self,
        mock_fin,
        mock_splits,
        mock_earnings,
        mock_div,
        mock_price,
    ):
        """Test message structure."""
        from src.data_ingestion.main import build_stock_message

        mock_price.return_value = {
            "Open": 150.0,
            "High": 155.0,
            "Low": 148.0,
            "Close": 152.0,
            "Volume": 5000000,
        }
        mock_div.return_value = 0.22
        mock_earnings.return_value = None
        mock_splits.return_value = None
        mock_fin.return_value = None

        result = build_stock_message("AAPL")

        assert result["ticker"] == "AAPL"
        assert "timestamp" in result
        assert "price" in result
        assert "dividends" in result
