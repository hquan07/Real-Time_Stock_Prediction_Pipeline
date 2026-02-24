from kafka import KafkaProducer
import json

from ..config.kafka_config import (
    KAFKA_BROKER,
    PRODUCER_CLIENT_ID
)


def create_kafka_producer():
    """
    Create Kafka producer instance
    """
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        client_id=PRODUCER_CLIENT_ID,
        acks="all",
        retries=5,
        linger_ms=10,
        value_serializer=lambda v: v  # bytes (Avro)
    )
    return producer