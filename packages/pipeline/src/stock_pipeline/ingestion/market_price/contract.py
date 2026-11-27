"""MarketPriceV1 construction, Avro wire format and semantic checks."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
import struct
from typing import Any

from fastavro import parse_schema, schemaless_reader, schemaless_writer
from fastavro.validation import validate as avro_validate

from stock_pipeline.ingestion.schema.contract import is_market_session, normalize_ticker
from .freshness import eod_is_stale


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "market_price_v1.avsc"
SCHEMA = json.loads(SCHEMA_PATH.read_text())
PARSED_SCHEMA = parse_schema(SCHEMA)
SCHEMA_VERSION = 1
SUBJECT = "market_price_raw-value"
VALIDATED_SUBJECT = "market_price_validated-value"
SOURCE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,49}$")
# The P1 research feed is EOD only. Minute/tick bars need a licensed provider
# and interval-specific calendar/completeness checks before being accepted.
INTERVALS = {"1d"}
WIRE_MAGIC = b"\x00"


def event_id_for(source: str, symbol: str, event_time_ms: int, interval: str) -> str:
    """Stable logical key, unaffected by ingestion time or a replay."""
    key = f"{source}|{normalize_ticker(symbol)}|{event_time_ms}|{interval}"
    return sha256(key.encode("utf-8")).hexdigest()


def make_event(
    *, source: str, symbol: str, event_time_ms: int, interval: str,
    open_price: float, high: float, low: float, close: float, volume: int,
    ingested_at_ms: int | None = None, quality_flags: list[str] | None = None,
) -> dict[str, Any]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    symbol = normalize_ticker(symbol)
    source = source.strip().lower()
    return {
        "event_id": event_id_for(source, symbol, event_time_ms, interval),
        "symbol": symbol,
        "event_time_utc": int(event_time_ms),
        "ingested_at_utc": int(ingested_at_ms if ingested_at_ms is not None else now_ms),
        "open": float(open_price), "high": float(high), "low": float(low),
        "close": float(close), "volume": int(volume),
        "source": source, "interval": interval,
        "exchange_timezone": "America/New_York",
        "schema_version": SCHEMA_VERSION,
        "quality_flags": list(quality_flags or []),
    }


def encode_event(event: dict[str, Any], schema_id: int) -> bytes:
    if schema_id <= 0:
        raise ValueError("A positive Schema Registry ID is required")
    stream = BytesIO()
    schemaless_writer(stream, PARSED_SCHEMA, event)
    return WIRE_MAGIC + struct.pack(">I", schema_id) + stream.getvalue()


def decode_event(payload: bytes, writer_schema: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    if len(payload) < 6 or payload[:1] != WIRE_MAGIC:
        raise ValueError("Not a Confluent Avro message")
    schema_id = struct.unpack(">I", payload[1:5])[0]
    writer = parse_schema(writer_schema) if writer_schema else PARSED_SCHEMA
    event = schemaless_reader(BytesIO(payload[5:]), writer, PARSED_SCHEMA)
    return schema_id, event


def check_event(event: dict[str, Any], *, now_ms: int | None = None,
                allow_historical: bool = False) -> list[str]:
    """Return stable check codes; invalid rows must be quarantined before Silver."""
    failures: list[str] = []
    try:
        if not avro_validate(event, PARSED_SCHEMA):
            failures.append("schema")
    except (TypeError, ValueError, KeyError):
        failures.append("schema")
    if failures:
        return failures

    try:
        symbol = normalize_ticker(event["symbol"])
    except (TypeError, ValueError):
        failures.append("symbol")
        symbol = None
    source = event["source"]
    interval = event["interval"]
    if not isinstance(source, str) or not SOURCE_PATTERN.fullmatch(source):
        failures.append("source")
    if interval not in INTERVALS:
        failures.append("interval")
    if event["schema_version"] != SCHEMA_VERSION:
        failures.append("schema_version")
    if event["exchange_timezone"] != "America/New_York":
        failures.append("exchange_timezone")

    event_ms = event["event_time_utc"]
    ingested_ms = event["ingested_at_utc"]
    clock_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    if event_ms <= 0 or ingested_ms < event_ms or event_ms > clock_ms + 300_000:
        failures.append("event_time")
    else:
        try:
            if not is_market_session(event_ms):
                failures.append("market_calendar")
        except (ValueError, OverflowError):
            failures.append("event_time")
    if not allow_historical and "event_time" not in failures:
        try:
            if eod_is_stale(
                datetime.fromtimestamp(event_ms / 1000, tz=timezone.utc),
                now=datetime.fromtimestamp(clock_ms / 1000, tz=timezone.utc),
            ):
                failures.append("freshness")
        except (ValueError, OverflowError):
            failures.append("freshness")

    prices = [event[name] for name in ("open", "high", "low", "close")]
    if any(not math.isfinite(value) or value <= 0 for value in prices):
        failures.append("price_range")
    elif event["low"] > min(event["open"], event["close"]) or event["high"] < max(event["open"], event["close"], event["low"]):
        failures.append("ohlc_consistency")
    if event["volume"] < 0:
        failures.append("volume_range")
    if symbol and isinstance(source, str) and SOURCE_PATTERN.fullmatch(source) and interval in INTERVALS:
        if event["event_id"] != event_id_for(source, symbol, event_ms, interval):
            failures.append("event_id")
    return failures
