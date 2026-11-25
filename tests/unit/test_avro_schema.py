import json
import pytest
from fastavro import parse_schema, validate
from unittest.mock import patch

from src.data_ingestion.main import build_stock_message


SCHEMA_PATH = "packages/pipeline/src/stock_pipeline/ingestion/schema/stock_schema.avsc"


@pytest.fixture(scope="module")
def avro_schema():
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return parse_schema(schema)


def test_message_matches_avro_schema(avro_schema):
    """Validate message against Avro schema"""
    price = {
        "open": 150.0,
        "high": 153.0,
        "low": 149.0,
        "close": 152.0,
        "volume": 1_000_000,
        "event_time_ms": 1795453200000,
    }
    with (
        patch("src.data_ingestion.main.fetch_price", return_value=price),
        patch("src.data_ingestion.main.fetch_dividends", return_value=None),
        patch("src.data_ingestion.main.fetch_earnings", return_value=None),
        patch("src.data_ingestion.main.fetch_splits", return_value=None),
        patch("src.data_ingestion.main.fetch_financials", return_value=None),
    ):
        message = build_stock_message("aapl")

    assert validate(message, avro_schema) is True
    assert message["ticker"] == "AAPL"
    assert len(message["event_id"]) == 64
    assert message["schema_version"] == 2


from fastavro.validation import ValidationError

def test_invalid_message_fails_schema(avro_schema):
    """Invalid message should not pass schema validation"""

    invalid_message = {
        "ticker": "AAPL"
        # missing required fields
    }

    with pytest.raises(ValidationError):
        validate(invalid_message, avro_schema)


def test_market_calendar_rejects_weekend():
    from datetime import datetime, timezone
    from src.data_ingestion.schema.contract import is_market_session, utc_millis

    saturday = datetime(2026, 11, 21, 15, 0, tzinfo=timezone.utc)
    assert is_market_session(utc_millis(saturday)) is False


def test_event_id_is_deterministic():
    from src.data_ingestion.schema.contract import build_event_id

    first = build_event_id("aapl", 1795453200000)
    second = build_event_id("AAPL", 1795453200000)
    assert first == second
