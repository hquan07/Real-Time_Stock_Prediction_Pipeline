import logging
from pathlib import Path
from fastavro import parse_schema, validate
import json
from .contract import (
    SCHEMA_VERSION,
    build_event_id,
    is_market_session,
    normalize_ticker,
    validate_ohlcv,
)

logger = logging.getLogger(__name__)

# Load schema once - schema is in the same directory as this file
SCHEMA_PATH = Path(__file__).resolve().parent / "stock_schema.avsc"

with open(SCHEMA_PATH, "r") as f:
    _schema = json.load(f)

PARSED_SCHEMA = parse_schema(_schema)


def validate_stock_message(message: dict) -> None:
    """
    Validate message against Avro schema.
    Raise ValueError if schema is invalid.
    """
    is_valid = validate(message, PARSED_SCHEMA)

    if not is_valid:
        logger.error("❌ Avro schema validation failed")
        raise ValueError("Invalid Avro schema for stock message")

    ticker = normalize_ticker(message["ticker"])
    if message["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {message['schema_version']}")
    if message["event_id"] != build_event_id(
        ticker, message["timestamp"], message["source"], message["interval"]
    ):
        raise ValueError("event_id does not match the canonical dedup key")
    if not is_market_session(message["timestamp"]):
        raise ValueError("Event timestamp is outside the configured market calendar")
    validate_ohlcv(message["price"])

    logger.debug("✔ Avro schema validation passed")
