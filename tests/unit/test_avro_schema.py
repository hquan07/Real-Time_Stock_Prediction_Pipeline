import json
import pytest
from fastavro import parse_schema, validate

from src.data_ingestion.main import build_stock_message


SCHEMA_PATH = "src/data_ingestion/schema/stock_schema.avsc"


@pytest.fixture(scope="module")
def avro_schema():
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return parse_schema(schema)


def test_message_matches_avro_schema(avro_schema):
    """Validate message against Avro schema"""

    message = build_stock_message("AAPL")

    assert validate(message, avro_schema) is True


def test_invalid_message_fails_schema(avro_schema):
    """Invalid message should not pass schema validation"""

    invalid_message = {
        "ticker": "AAPL"
        # missing required fields
    }

    assert validate(invalid_message, avro_schema) is False