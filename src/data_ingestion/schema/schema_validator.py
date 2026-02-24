import logging
from pathlib import Path
from fastavro import parse_schema, validate
import json

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

    logger.debug("✔ Avro schema validation passed")