"""Publish the latest completed XNYS daily bar for the research universe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from uuid import uuid4

from kafka import KafkaProducer
import pandas as pd
import yfinance as yf

from stock_pipeline.config.settings import KafkaSettings
from stock_pipeline.ingestion.fetcher.tickers import get_ticker_list

from .contract import check_event, encode_event, make_event
from .freshness import expected_eod_session
from .registry import SchemaRegistry
from .replay import DeliveryStats, _ensure_topics, send_with_ack
from .store import RAW_TOPIC


logger = logging.getLogger(__name__)


def fetch_completed_bars(tickers: list[str], *, now: datetime | None = None) -> list[dict]:
    """Require one complete bar per configured symbol before publishing any."""
    now = now or datetime.now(timezone.utc)
    expected = expected_eod_session(now)
    now_ms = int(now.timestamp() * 1000)
    events = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(
            period="10d", interval="1d", auto_adjust=False, actions=False,
        )
        if history.empty:
            raise ValueError(f"No daily history for {ticker}")
        matching = []
        for index, row in history.iterrows():
            timestamp = pd.Timestamp(index)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("America/New_York")
            if timestamp.tz_convert("America/New_York").date() == expected:
                matching.append((timestamp, row))
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one completed {expected} bar for {ticker}")
        timestamp, row = matching[0]
        event = make_event(
            source="yfinance", symbol=ticker,
            event_time_ms=int(timestamp.tz_convert("UTC").timestamp() * 1000),
            interval="1d", open_price=float(row["Open"]),
            high=float(row["High"]), low=float(row["Low"]),
            close=float(row["Close"]), volume=int(row["Volume"]),
            ingested_at_ms=now_ms,
        )
        failures = check_event(event, now_ms=now_ms)
        if failures:
            raise ValueError(f"Invalid daily bar for {ticker}: {', '.join(failures)}")
        events.append(event)
    return events


def publish_latest(tickers: list[str]) -> tuple[str, int]:
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Ticker universe must be nonempty and unique")
    events = fetch_completed_bars(tickers)
    settings = KafkaSettings.from_env()
    _ensure_topics(settings.bootstrap_servers)
    schema_id = SchemaRegistry(
        os.getenv("KAFKA_SCHEMA_REGISTRY_URL", "http://schema-registry:8081")
    ).register_market_price()
    run_id = uuid4().hex
    headers = [("run_id", run_id.encode("ascii")),
               ("expected_records", str(len(events)).encode("ascii")),
               ("historical", b"0")]
    producer = KafkaProducer(
        bootstrap_servers=settings.bootstrap_servers, acks="all", retries=3,
        retry_backoff_ms=500, max_block_ms=10_000,
    )
    stats = DeliveryStats()
    try:
        for event in events:
            send_with_ack(
                producer, topic=RAW_TOPIC, key=event["symbol"].encode("utf-8"),
                value=encode_event(event, schema_id), headers=headers, stats=stats,
            )
    finally:
        producer.flush(timeout=15)
        producer.close()
        logger.info("EOD delivery run_id=%s delivered=%s failed=%s",
                    run_id, stats.delivered, stats.failed)
    return run_id, stats.delivered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", help="Comma-separated research universe")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tickers = ([ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
               if args.tickers else get_ticker_list())
    run_id, count = publish_latest(tickers)
    print(f"run_id={run_id} published={count}")


if __name__ == "__main__":
    main()
