"""Live P1 storage/replay drill against an isolated PostgreSQL database.

The caller must set P1_TEST_DATABASE to the same disposable database as
POSTGRES_DB. This script refuses the normal stockdb database.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os

import psycopg2

from stock_pipeline.config.settings import DatabaseSettings
from stock_pipeline.ingestion.market_price.contract import SCHEMA, encode_event, make_event
from stock_pipeline.ingestion.market_price.store import MarketPriceStore, RAW_TOPIC


class StaticRegistry:
    def get_schema(self, schema_id: int) -> dict:
        if schema_id != 1:
            raise ValueError("Unknown Schema Registry ID")
        return SCHEMA


def _ms(day: int) -> int:
    return int(datetime(2024, 1, day, 21, tzinfo=timezone.utc).timestamp() * 1000)


def _event(day: int, *, close: float = 101.0, symbol: str = "AAPL") -> bytes:
    event = make_event(
        source="yfinance", symbol=symbol, event_time_ms=_ms(day),
        interval="1d", open_price=100.0, high=105.0, low=99.0,
        close=close, volume=1000,
    )
    return encode_event(event, 1)


def _counts(connection) -> tuple[int, int, int, int, int]:
    with connection.cursor() as cursor:
        cursor.execute("""SELECT
          (SELECT count(*) FROM public.market_price_bronze),
          (SELECT count(*) FROM public.market_price_silver),
          (SELECT count(*) FROM public.market_price_gold_bars),
          (SELECT count(*) FROM public.market_price_quarantine),
          (SELECT count(*) FROM public.market_price_outbox)""")
        return cursor.fetchone()


def main() -> None:
    settings = DatabaseSettings.from_env()
    if (settings.database == "stockdb" or not os.getenv("P1_TEST_DATABASE")
            or settings.database != os.environ["P1_TEST_DATABASE"]):
        raise RuntimeError("Refusing P1 storage drill outside the named disposable database")

    store = MarketPriceStore(StaticRegistry())
    first = _event(2)
    second = _event(3)
    assert store.process(first, topic=RAW_TOPIC, partition=0, offset=0,
                         run_id="p1-test-good", expected_records=2,
                         allow_historical=True) == "accepted"
    assert store.process(second, topic=RAW_TOPIC, partition=0, offset=1,
                         run_id="p1-test-good", expected_records=2,
                         allow_historical=True) == "accepted"
    assert store.process(second, topic=RAW_TOPIC, partition=0, offset=1,
                         run_id="p1-test-good", expected_records=2,
                         allow_historical=True) == "offset_duplicate"
    with psycopg2.connect(**settings.connect_kwargs()) as connection:
        assert _counts(connection) == (2, 2, 2, 0, 2)
        with connection.cursor() as cursor:
            cursor.execute("""SELECT count(*) FROM public.market_price_gold_bars g
                              JOIN public.data_quality_runs q ON q.run_id = g.quality_run_id
                              WHERE g.as_of_time < q.finished_at""")
            assert cursor.fetchone() == (0,)

    # Same file on new offsets: Bronze preserves both deliveries, logical
    # Silver/Gold counts and validated publication remain unchanged.
    assert store.process(first, topic=RAW_TOPIC, partition=0, offset=2,
                         run_id="p1-test-replay", expected_records=2,
                         allow_historical=True) == "duplicate"
    assert store.process(second, topic=RAW_TOPIC, partition=0, offset=3,
                         run_id="p1-test-replay", expected_records=2,
                         allow_historical=True) == "duplicate"
    with psycopg2.connect(**settings.connect_kwargs()) as connection:
        assert _counts(connection) == (4, 2, 2, 0, 2)

    # A failed run may retain an inspected Silver row but cannot publish Gold.
    assert store.process(_event(4), topic=RAW_TOPIC, partition=0, offset=4,
                         run_id="p1-test-bad", expected_records=2,
                         allow_historical=True) == "accepted"
    assert store.process(_event(5, close=-1), topic=RAW_TOPIC, partition=0, offset=5,
                         run_id="p1-test-bad", expected_records=2,
                         allow_historical=True) == "rejected"
    with psycopg2.connect(**settings.connect_kwargs()) as connection:
        assert _counts(connection) == (6, 3, 2, 1, 3)
        with connection.cursor() as cursor:
            cursor.execute("SELECT status, rejected_records FROM public.data_quality_runs WHERE run_id = 'p1-test-bad'")
            assert cursor.fetchone() == ("failed", 1)
            cursor.execute("SELECT event_id FROM public.market_price_gold_bars WHERE quality_run_id = 'p1-test-bad'")
            assert cursor.fetchone() is None
            cursor.execute("""SELECT g.event_id, s.event_id, b.id
                              FROM public.market_price_gold_bars g
                              JOIN public.market_price_silver s USING (event_id)
                              JOIN public.market_price_bronze b ON b.id = g.bronze_id
                              LIMIT 1""")
            assert cursor.fetchone() is not None

    # Out-of-order arrival must not hide a missing trading session.
    assert store.process(_event(9, symbol="MSFT"), topic=RAW_TOPIC, partition=1,
                         offset=0, run_id="p1-test-gap", expected_records=2,
                         allow_historical=True) == "accepted"
    assert store.process(_event(5, symbol="MSFT"), topic=RAW_TOPIC, partition=1,
                         offset=1, run_id="p1-test-gap", expected_records=2,
                         allow_historical=True) == "accepted"
    with psycopg2.connect(**settings.connect_kwargs()) as connection:
        assert _counts(connection) == (8, 5, 2, 1, 3)
        with connection.cursor() as cursor:
            cursor.execute("SELECT status FROM public.data_quality_runs WHERE run_id = 'p1-test-gap'")
            assert cursor.fetchone() == ("failed",)
            cursor.execute("""SELECT count(*) FROM public.data_quality_results
                              WHERE run_id = 'p1-test-gap' AND check_code = 'missing_bars'
                                AND NOT passed""")
            assert cursor.fetchone() == (1,)
    bad_schema = bytearray(_event(10, symbol="NVDA"))
    bad_schema[1:5] = (999).to_bytes(4, "big")
    assert store.process(bytes(bad_schema), topic=RAW_TOPIC, partition=2, offset=0,
                         run_id="p1-test-schema", expected_records=1,
                         allow_historical=True) == "rejected"
    with psycopg2.connect(**settings.connect_kwargs()) as connection:
        assert _counts(connection) == (9, 5, 2, 2, 4)
    print("P1 storage drill passed: replay deduplicated; invalid, missing-bar and unknown-schema runs blocked from Gold")


if __name__ == "__main__":
    main()
