import os
from .settings import KafkaSettings

KAFKA_BROKER = KafkaSettings.from_env().bootstrap_servers
KAFKA_SCHEMA_REGISTRY = os.getenv("KAFKA_SCHEMA_REGISTRY", "http://schema-registry:8081")

TOPIC_STOCK_RAW = os.getenv("TOPIC_STOCK_RAW", "stock_raw_data")
TOPIC_STOCK_FEATURES = os.getenv("TOPIC_STOCK_FEATURES", "stock_features")
TOPIC_STOCK_PREDICTION = os.getenv("TOPIC_STOCK_PREDICTION", "stock_prediction")
