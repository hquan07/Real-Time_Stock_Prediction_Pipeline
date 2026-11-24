import os
from config.settings import KafkaSettings

_settings = KafkaSettings.from_env()
KAFKA_BROKER = _settings.bootstrap_servers
KAFKA_TOPIC = _settings.topic
AVRO_SCHEMA_PATH = os.getenv("AVRO_SCHEMA_PATH", str(_settings.schema_path))

PRODUCER_CLIENT_ID = os.getenv(
    "PRODUCER_CLIENT_ID",
    "yfinance-producer"
)
