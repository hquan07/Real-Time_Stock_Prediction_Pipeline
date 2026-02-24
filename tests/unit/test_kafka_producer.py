import pytest
from unittest.mock import patch, MagicMock

from src.data_ingestion.main import build_stock_message


def test_build_stock_message_structure():
    """Test message structure for Kafka producer"""

    message = build_stock_message("AAPL")

    assert message["ticker"] == "AAPL"
    assert "timestamp" in message

    assert "price" in message
    assert "dividends" in message
    assert "earnings" in message
    assert "financials" in message


@patch("src.data_ingestion.kafka_producer.send_data.send_to_kafka")
def test_send_to_kafka_called(mock_send):
    """Ensure send_to_kafka is called correctly"""

    message = build_stock_message("MSFT")

    mock_send(topic="stock_raw_data", data=message)

    mock_send.assert_called_once()
    args, kwargs = mock_send.call_args

    assert kwargs["topic"] == "stock_raw_data"
    assert kwargs["data"]["ticker"] == "MSFT"