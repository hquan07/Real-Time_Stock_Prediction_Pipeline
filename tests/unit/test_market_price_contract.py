"""P1 contract compatibility and data-quality boundaries."""

from datetime import datetime, timezone
from io import BytesIO
import struct

from fastavro import parse_schema, schemaless_writer

from stock_pipeline.ingestion.market_price.contract import (
    SCHEMA, check_event, decode_event, encode_event, make_event,
)


def _ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, 21, tzinfo=timezone.utc).timestamp() * 1000)


def _event(**changes):
    event = make_event(
        source="fixture", symbol="aapl", event_time_ms=_ms(2024, 1, 2),
        interval="1d", open_price=100, high=105, low=99, close=101,
        volume=1000, ingested_at_ms=_ms(2024, 1, 3),
    )
    event.update(changes)
    return event


def test_market_price_wire_roundtrip_and_deterministic_id():
    event = _event()
    assert event["symbol"] == "AAPL"
    assert event["event_id"] == _event(ingested_at_utc=_ms(2024, 1, 4))["event_id"]
    schema_id, decoded = decode_event(encode_event(event, 42), SCHEMA)
    assert schema_id == 42
    assert decoded == event
    assert check_event(decoded, allow_historical=True) == []
    assert "earnings" not in [field["name"] for field in SCHEMA["fields"]]


def test_backward_reader_accepts_added_defaulted_field():
    old_writer = {**SCHEMA, "fields": [
        field for field in SCHEMA["fields"] if field["name"] != "quality_flags"
    ]}
    event = _event()
    stream = BytesIO()
    schemaless_writer(stream, parse_schema(old_writer), event)
    wire = b"\x00" + struct.pack(">I", 7) + stream.getvalue()
    _, decoded = decode_event(wire, old_writer)
    assert decoded["quality_flags"] == []


def test_invalid_price_event_id_calendar_and_freshness():
    event = _event(close=-1)
    assert "price_range" in check_event(event, allow_historical=True)
    event = _event(high=100)
    assert "ohlc_consistency" in check_event(event, allow_historical=True)
    event = _event(event_id="0" * 64)
    assert "event_id" in check_event(event, allow_historical=True)
    event = _event(event_time_utc=_ms(2024, 1, 1))
    assert "market_calendar" in check_event(event, allow_historical=True)
    event = _event()
    assert "freshness" in check_event(event, now_ms=_ms(2024, 1, 10))
    assert "freshness" not in check_event(event, allow_historical=True)


def test_interval_without_quality_policy_is_rejected():
    assert "interval" in check_event(_event(interval="1m"), allow_historical=True)
