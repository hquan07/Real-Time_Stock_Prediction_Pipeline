from .producer import create_kafka_producer
from .avro_serializer import AvroSerializer
from ..config.kafka_config import KAFKA_TOPIC


_producer = None
_serializer = None


def _init():
    global _producer, _serializer
    if _producer is None:
        _producer = create_kafka_producer()
    if _serializer is None:
        _serializer = AvroSerializer()


def send_to_kafka(topic: str = KAFKA_TOPIC, data: dict = None):
    """
    Serialize data with Avro and send to Kafka
    """
    if data is None:
        return

    _init()

    try:
        avro_bytes = _serializer.serialize(data)

        future = _producer.send(
            topic=topic,
            value=avro_bytes,
            key=data["ticker"].encode("utf-8")
        )

        future.get(timeout=10)

    except Exception as e:
        print(f"❌ Kafka send failed: {e}")