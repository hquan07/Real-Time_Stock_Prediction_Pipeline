"""Kafka raw consumer with durable DB/outbox acknowledgement ordering."""

from __future__ import annotations

import logging
import os

from kafka import KafkaConsumer, KafkaProducer, OffsetAndMetadata, TopicPartition

from stock_pipeline.config.settings import KafkaSettings

from .registry import SchemaRegistry
from .store import MarketPriceStore, RAW_TOPIC


logger = logging.getLogger(__name__)


def run() -> None:
    kafka = KafkaSettings.from_env()
    registry = SchemaRegistry(os.getenv("KAFKA_SCHEMA_REGISTRY_URL", "http://schema-registry:8081"))
    store = MarketPriceStore(registry)
    producer = KafkaProducer(
        bootstrap_servers=kafka.bootstrap_servers, acks="all", retries=3,
        retry_backoff_ms=500, max_block_ms=10_000,
    )
    consumer = KafkaConsumer(
        os.getenv("MARKET_PRICE_RAW_TOPIC", RAW_TOPIC),
        bootstrap_servers=kafka.bootstrap_servers,
        group_id="p1-market-price-v1", enable_auto_commit=False,
        auto_offset_reset="earliest", max_poll_records=100,
    )
    try:
        while store.publish_outbox(producer):
            pass
        for record in consumer:
            headers = dict(record.headers or [])
            run_id = headers.get(b"run_id") or headers.get("run_id")
            expected = headers.get(b"expected_records") or headers.get("expected_records")
            historical = headers.get(b"historical") or headers.get("historical")
            run_id = run_id.decode("ascii") if run_id else f"live-{record.topic}-{record.partition}-{record.offset}"
            expected_records = int(expected.decode("ascii")) if expected else 1
            allow_historical = historical == b"1"
            outcome = store.process(
                record.value, topic=record.topic, partition=record.partition,
                offset=record.offset, run_id=run_id,
                expected_records=expected_records,
                allow_historical=allow_historical,
            )
            # Commit only this processed partition/offset. A bare commit()
            # could advance past other records already prefetched by poll().
            consumer.commit({
                TopicPartition(record.topic, record.partition):
                    OffsetAndMetadata(record.offset + 1, "", -1),
            })
            store.publish_outbox(producer)
            logger.info("market_price record=%s partition=%s offset=%s result=%s",
                        record.topic, record.partition, record.offset, outcome)
    finally:
        consumer.close()
        producer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
