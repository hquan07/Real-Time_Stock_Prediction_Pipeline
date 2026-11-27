"""Publish a CSV of EOD US-equity bars using MarketPriceV1."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError

from stock_pipeline.config.settings import KafkaSettings

from .contract import encode_event, make_event
from .registry import SchemaRegistry
from .store import DLQ_TOPIC, RAW_TOPIC, VALIDATED_TOPIC


logger = logging.getLogger(__name__)


@dataclass
class DeliveryStats:
    delivered: int = 0
    failed: int = 0

    def on_success(self, _metadata) -> None:
        self.delivered += 1

    def on_error(self, error) -> None:
        self.failed += 1
        logger.error("Kafka delivery failed: %s", type(error).__name__)


def send_with_ack(producer, *, topic: str, key: bytes, value: bytes,
                  headers: list[tuple[str, bytes]], stats: DeliveryStats) -> None:
    future = producer.send(topic, key=key, value=value, headers=headers)
    future.add_callback(stats.on_success)
    future.add_errback(stats.on_error)
    future.get(timeout=15)


def _timestamp_ms(value: str) -> int:
    if value.isdecimal():
        return int(value)
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("event_time_utc must include a timezone")
    return int(timestamp.astimezone(timezone.utc).timestamp() * 1000)


def _ensure_topics(bootstrap_servers: str) -> None:
    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    try:
        topics = [
            NewTopic(RAW_TOPIC, 3, 1, topic_configs={"retention.ms": "604800000"}),
            NewTopic(VALIDATED_TOPIC, 3, 1, topic_configs={"retention.ms": "604800000"}),
            NewTopic(DLQ_TOPIC, 3, 1, topic_configs={"retention.ms": "2592000000"}),
        ]
        existing = set(admin.list_topics())
        missing = [topic for topic in topics if topic.name not in existing]
        if missing:
            try:
                admin.create_topics(missing, validate_only=False)
            except TopicAlreadyExistsError:
                pass
    finally:
        admin.close()


def publish_csv(path: Path, *, source: str, historical: bool = False) -> tuple[str, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("CSV has no data rows")
    if len(rows) > 100_000:
        raise ValueError("Replay accepts at most 100000 rows per run")

    kafka = KafkaSettings.from_env()
    _ensure_topics(kafka.bootstrap_servers)
    schema_id = SchemaRegistry(
        os.getenv("KAFKA_SCHEMA_REGISTRY_URL", "http://schema-registry:8081")
    ).register_market_price()
    run_id = uuid4().hex
    headers = [
        ("run_id", run_id.encode("ascii")),
        ("expected_records", str(len(rows)).encode("ascii")),
        ("historical", b"1" if historical else b"0"),
    ]
    producer = KafkaProducer(
        bootstrap_servers=kafka.bootstrap_servers, acks="all", retries=3,
        retry_backoff_ms=500, max_block_ms=10_000,
    )
    stats = DeliveryStats()
    attempted = 0
    try:
        for row in rows:
            symbol = (row.get("symbol") or "unknown").strip().upper()
            try:
                event = make_event(
                    source=source, symbol=symbol,
                    event_time_ms=_timestamp_ms(row["event_time_utc"]),
                    interval=row.get("interval") or "1d",
                    open_price=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                payload = encode_event(event, schema_id)
                symbol = event["symbol"]
            except (KeyError, TypeError, ValueError) as error:
                logger.warning("row %s failed local encoding: %s; sending to quarantine path",
                               attempted + 1, type(error).__name__)
                payload = json.dumps({"invalid_csv_row": row}, separators=(",", ":")).encode("utf-8")
            attempted += 1
            send_with_ack(producer, topic=RAW_TOPIC, key=symbol.encode("utf-8"),
                          value=payload, headers=headers, stats=stats)
    finally:
        producer.flush(timeout=15)
        producer.close()
        logger.info("delivery run_id=%s attempted=%s delivered=%s failed=%s",
                    run_id, attempted, stats.delivered, stats.failed)
    return run_id, stats.delivered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--historical", action="store_true",
                        help="Allow old bars for controlled research replay")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_id, count = publish_csv(args.file, source=args.source,
                                historical=args.historical)
    print(f"run_id={run_id} published={count}")


if __name__ == "__main__":
    main()
