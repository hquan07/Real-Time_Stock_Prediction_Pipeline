"""
Unit tests for database connection module.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestDBConnection:
    """Tests for database connection functions."""

    def test_database_url_construction(self):
        """Test that database URL is constructed correctly."""
        with patch.dict("os.environ", {
            "DB_HOST": "testhost",
            "DB_PORT": "5433",
            "DB_NAME": "testdb",
            "DB_USER": "testuser",
            "DB_PASSWORD": "testpass",
        }):
            # Re-import to pick up new env vars
            import importlib
            import src.database.db_connection as db_module
            importlib.reload(db_module)

            expected = "postgresql://testuser:testpass@testhost:5433/testdb"
            assert db_module.DATABASE_URL == expected

    @patch("src.database.db_connection.create_engine")
    def test_get_engine_creates_engine(self, mock_create_engine):
        """Test that get_engine creates engine on first call."""
        from src.database.db_connection import get_engine, dispose_engine

        # Clear any existing engine
        dispose_engine()

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        engine = get_engine()

        assert mock_create_engine.called
        assert engine == mock_engine

    @patch("src.database.db_connection.create_engine")
    def test_get_engine_returns_cached(self, mock_create_engine):
        """Test that get_engine returns cached engine on subsequent calls."""
        from src.database.db_connection import get_engine, dispose_engine

        dispose_engine()

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        engine1 = get_engine()
        engine2 = get_engine()

        # Should only be called once
        assert mock_create_engine.call_count == 1
        assert engine1 is engine2

    @patch("src.database.db_connection.get_engine")
    def test_check_connection_success(self, mock_get_engine):
        """Test check_connection returns True on success."""
        from src.database.db_connection import check_connection

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_engine.return_value = mock_engine

        result = check_connection()

        assert result is True

    @patch("src.database.db_connection.get_engine")
    def test_check_connection_failure(self, mock_get_engine):
        """Test check_connection returns False on failure."""
        from src.database.db_connection import check_connection

        mock_get_engine.side_effect = Exception("Connection failed")

        result = check_connection()

        assert result is False


class TestWriteRaw:
    """Tests for write_raw module."""

    def test_write_company_success(self, mock_db_session):
        """Test successful company write."""
        from src.database.write_raw import write_company

        result = write_company(
            ticker="AAPL",
            long_name="Apple Inc.",
            sector="Technology",
        )

        assert result is True

    def test_write_price_history_success(self, mock_db_session):
        """Test successful price history write."""
        from src.database.write_raw import write_price_history
        from datetime import date

        result = write_price_history(
            ticker="AAPL",
            date_val=date.today(),
            open_price=150.0,
            high=155.0,
            low=148.0,
            close=152.0,
            volume=5000000,
        )

        assert result is True

    def test_write_kafka_message(self, mock_db_session, sample_kafka_message):
        """Test writing Kafka message to database."""
        from src.database.write_raw import write_kafka_message

        result = write_kafka_message(sample_kafka_message)

        assert result is True

    def test_write_kafka_message_missing_ticker(self, mock_db_session):
        """Test handling message without ticker."""
        from src.database.write_raw import write_kafka_message

        result = write_kafka_message({"timestamp": 1234567890})

        assert result is False


class TestWritePrediction:
    """Tests for write_prediction module."""

    def test_write_prediction_success(self, mock_db_session):
        """Test successful prediction write."""
        from src.database.write_prediction import write_prediction
        from datetime import date

        result = write_prediction(
            ticker="AAPL",
            prediction_date=date.today(),
            target_date=date.today(),
            model_name="RandomForest",
            predicted_close=155.0,
            confidence=0.85,
        )

        assert result is True

    def test_write_batch_predictions(self, mock_db_session):
        """Test batch prediction write."""
        from src.database.write_prediction import write_batch_predictions
        from datetime import date

        predictions = [
            {
                "ticker": "AAPL",
                "prediction_date": date.today(),
                "target_date": date.today(),
                "model_name": "RandomForest",
                "predicted_close": 155.0,
            },
            {
                "ticker": "MSFT",
                "prediction_date": date.today(),
                "target_date": date.today(),
                "model_name": "RandomForest",
                "predicted_close": 305.0,
            },
        ]

        count = write_batch_predictions(predictions)

        assert count == 2
