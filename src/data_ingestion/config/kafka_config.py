import os
from pathlib import Path

# Default to localhost for local development, override via env for Docker
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "stock_raw_data")

# Calculate path relative to this file
_CONFIG_DIR = Path(__file__).parent
_SCHEMA_DIR = _CONFIG_DIR.parent / "schema"

AVRO_SCHEMA_PATH = os.getenv(
    "AVRO_SCHEMA_PATH",
    str(_SCHEMA_DIR / "stock_schema.avsc")
)

PRODUCER_CLIENT_ID = os.getenv(
    "PRODUCER_CLIENT_ID",
    "yfinance-producer"
)