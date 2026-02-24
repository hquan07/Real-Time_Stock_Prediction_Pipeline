import os

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_SCHEMA_REGISTRY = os.getenv("KAFKA_SCHEMA_REGISTRY", "http://schema-registry:8081")

TOPIC_STOCK_RAW = os.getenv("TOPIC_STOCK_RAW", "stock_raw_data")
TOPIC_STOCK_FEATURES = os.getenv("TOPIC_STOCK_FEATURES", "stock_features")
TOPIC_STOCK_PREDICTION = os.getenv("TOPIC_STOCK_PREDICTION", "stock_prediction")