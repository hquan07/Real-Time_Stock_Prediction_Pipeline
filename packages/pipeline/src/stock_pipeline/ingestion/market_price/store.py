"""Transactional Bronze/Silver processing and run-gated Gold publication."""

from __future__ import annotations

import base64
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import struct
from typing import Callable

import exchange_calendars as xcals
import pandas as pd
import psycopg2

from stock_pipeline.config.settings import DatabaseSettings

from .contract import check_event, decode_event
from .registry import SchemaRegistry


RAW_TOPIC = "market_price_raw"
VALIDATED_TOPIC = "market_price_validated"
DLQ_TOPIC = "market_price_dlq"


def _utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _sequence_failures(event: dict, latest: datetime | None, *,
                       allow_historical: bool) -> list[str]:
    if latest is None:
        return []
    current = _utc(event["event_time_utc"])
    if current < latest:
        if not allow_historical and (latest - current).days > 7:
            return ["late_arrival"]
        return []
    if current == latest or event["interval"] != "1d":
        return []
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(pd.Timestamp(latest.date()), pd.Timestamp(current.date()))
    return ["missing_bars"] if len(sessions) > 2 else []


class MarketPriceStore:
    def __init__(self, registry: SchemaRegistry,
                 connect: Callable | None = None):
        self.registry = registry
        self.connect = connect or (lambda: psycopg2.connect(
            **DatabaseSettings.from_env().connect_kwargs()))

    @staticmethod
    def _run_has_missing_bars(cursor, run_id: str) -> bool:
        """Catch out-of-order file delivery that masked a per-record gap."""
        cursor.execute(
            """SELECT source, symbol, interval, event_time_utc,
                      bronze_id, event_id
               FROM public.market_price_silver
               WHERE quality_run_id = %s
               ORDER BY source, symbol, interval, event_time_utc""",
            (run_id,),
        )
        previous: dict[tuple[str, str, str], datetime] = {}
        missing = False
        for source, symbol, interval, event_time, bronze_id, event_id in cursor.fetchall():
            key = (source, symbol, interval)
            prior = previous.get(key)
            if prior is not None and _sequence_failures(
                {"event_time_utc": int(event_time.timestamp() * 1000),
                 "interval": interval}, prior, allow_historical=True,
            ):
                cursor.execute(
                    """INSERT INTO public.data_quality_results
                       (run_id, bronze_id, event_id, check_code, passed)
                       VALUES (%s, %s, %s, 'missing_bars', false)
                       ON CONFLICT (bronze_id, check_code) DO NOTHING""",
                    (run_id, bronze_id, event_id),
                )
                missing = True
            previous[key] = event_time
        return missing

    def process(
        self, payload: bytes, *, topic: str, partition: int, offset: int,
        run_id: str, expected_records: int = 1,
        allow_historical: bool = False,
    ) -> str:
        """Process one Kafka record. A repeated offset changes no database row."""
        if not (1 <= expected_records <= 100_000):
            raise ValueError("expected_records must be between 1 and 100000")
        if not run_id or len(run_id) > 128:
            raise ValueError("run_id must be 1..128 characters")
        if len(payload) > 1_048_576:
            raise ValueError("Raw event exceeds 1 MiB")

        schema_id = struct.unpack(">I", payload[1:5])[0] if len(payload) >= 5 and payload[:1] == b"\x00" else None
        with closing(self.connect()) as connection, connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT id FROM public.market_price_bronze
                       WHERE kafka_topic = %s AND kafka_partition = %s
                         AND kafka_offset = %s""",
                    (topic, partition, offset),
                )
                if cursor.fetchone() is not None:
                    return "offset_duplicate"
                cursor.execute(
                    """INSERT INTO public.data_quality_runs (run_id, expected_records)
                       VALUES (%s, %s)
                       ON CONFLICT (run_id) DO UPDATE
                       SET expected_records = EXCLUDED.expected_records
                       WHERE data_quality_runs.expected_records = EXCLUDED.expected_records
                       RETURNING processed_records, accepted_records, rejected_records,
                                 duplicate_records, status""",
                    (run_id, expected_records),
                )
                run = cursor.fetchone()
                if run is None:
                    raise ValueError("run_id reused with a different expected_records")
                if run[4] != "running":
                    raise ValueError("run_id already finalized; use a new ID for replay")
                cursor.execute(
                    """INSERT INTO public.market_price_bronze
                       (kafka_topic, kafka_partition, kafka_offset, payload,
                        payload_sha256, schema_id, run_id)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (kafka_topic, kafka_partition, kafka_offset)
                       DO NOTHING RETURNING id""",
                    (topic, partition, offset, psycopg2.Binary(payload),
                     sha256(payload).hexdigest(), schema_id, run_id),
                )
                row = cursor.fetchone()
                if row is None:
                    return "offset_duplicate"
                bronze_id = row[0]

                event = None
                failures: list[str] = []
                try:
                    if schema_id is None:
                        raise ValueError("Missing Schema Registry framing")
                    writer_schema = self.registry.get_schema(schema_id)
                    _, event = decode_event(payload, writer_schema)
                    failures = check_event(event, allow_historical=allow_historical)
                except (ValueError, KeyError, TypeError, OverflowError, EOFError):
                    failures = ["decode"]

                if event is not None and not failures:
                    cursor.execute(
                        """SELECT event_time_utc FROM public.market_price_silver
                           WHERE source = %s AND symbol = %s AND interval = %s
                           ORDER BY event_time_utc DESC LIMIT 1""",
                        (event["source"], event["symbol"], event["interval"]),
                    )
                    latest_row = cursor.fetchone()
                    failures.extend(_sequence_failures(
                        event, latest_row[0] if latest_row else None,
                        allow_historical=allow_historical,
                    ))

                event_id = event.get("event_id") if event else None
                duplicate = False
                if not failures and event is not None:
                    cursor.execute(
                        """INSERT INTO public.market_price_silver
                           (event_id, source, symbol, interval, event_time_utc,
                            ingested_at_utc, open, high, low, close, volume,
                            exchange_timezone, schema_version, bronze_id, quality_run_id)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                   %s, %s, %s, %s, %s)
                           ON CONFLICT DO NOTHING RETURNING event_id""",
                        (event_id, event["source"], event["symbol"], event["interval"],
                         _utc(event["event_time_utc"]), _utc(event["ingested_at_utc"]),
                         event["open"], event["high"], event["low"], event["close"],
                         event["volume"], event["exchange_timezone"],
                         event["schema_version"], bronze_id, run_id),
                    )
                    if cursor.fetchone() is None:
                        cursor.execute(
                            """SELECT event_id, open, high, low, close, volume
                               FROM public.market_price_silver WHERE event_id = %s""",
                            (event_id,),
                        )
                        existing = cursor.fetchone()
                        if existing is None or tuple(existing[1:]) != tuple(
                            event[key] for key in ("open", "high", "low", "close", "volume")
                        ):
                            failures.append("conflicting_duplicate")
                        else:
                            duplicate = True
                            cursor.execute(
                                """UPDATE public.market_price_silver
                                   SET quality_run_id = %s WHERE event_id = %s""",
                                (run_id, event_id),
                            )

                if failures:
                    for code in sorted(set(failures)):
                        cursor.execute(
                            """INSERT INTO public.data_quality_results
                               (run_id, bronze_id, event_id, check_code, passed)
                               VALUES (%s, %s, %s, %s, false)""",
                            (run_id, bronze_id, event_id, code),
                        )
                    cursor.execute(
                        """INSERT INTO public.market_price_quarantine
                           (bronze_id, reason_codes) VALUES (%s, %s)""",
                        (bronze_id, sorted(set(failures))),
                    )
                    dlq_payload = json.dumps({
                        "run_id": run_id, "reason_codes": sorted(set(failures)),
                        "raw_payload_base64": base64.b64encode(payload).decode("ascii"),
                    }, separators=(",", ":")).encode("utf-8")
                    cursor.execute(
                        """INSERT INTO public.market_price_outbox
                           (bronze_id, topic, message_key, payload)
                           VALUES (%s, %s, %s, %s)""",
                        (bronze_id, DLQ_TOPIC, (event.get("symbol") if event else "unknown").encode(),
                         psycopg2.Binary(dlq_payload)),
                    )
                    result = "rejected"
                else:
                    code = "duplicate" if duplicate else "contract"
                    cursor.execute(
                        """INSERT INTO public.data_quality_results
                           (run_id, bronze_id, event_id, check_code, passed)
                           VALUES (%s, %s, %s, %s, true)""",
                        (run_id, bronze_id, event_id, code),
                    )
                    result = "duplicate" if duplicate else "accepted"

                accepted = run[1] + (result == "accepted")
                rejected = run[2] + (result == "rejected")
                duplicates = run[3] + (result == "duplicate")
                processed = run[0] + 1
                status = "running"
                if processed == expected_records:
                    status = "failed" if rejected else "passed"
                    if status == "passed" and self._run_has_missing_bars(cursor, run_id):
                        status = "failed"
                cursor.execute(
                    """UPDATE public.data_quality_runs
                       SET processed_records = %s, accepted_records = %s,
                           rejected_records = %s, duplicate_records = %s,
                           status = %s,
                           finished_at = CASE WHEN %s = 'running' THEN NULL
                                              ELSE CURRENT_TIMESTAMP END
                       WHERE run_id = %s""",
                    (processed, accepted, rejected, duplicates, status, status, run_id),
                )
                if status == "passed":
                    cursor.execute(
                        """INSERT INTO public.market_price_gold_bars
                           (event_id, source, symbol, interval, event_time_utc,
                            as_of_time, open, high, low, close, volume,
                            bronze_id, quality_run_id)
                           SELECT s.event_id, s.source, s.symbol, s.interval,
                                  s.event_time_utc,
                                  GREATEST(b.received_at, q.finished_at), s.open,
                                  s.high, s.low, s.close, s.volume,
                                  s.bronze_id, s.quality_run_id
                           FROM public.market_price_silver s
                           JOIN public.market_price_bronze b ON b.id = s.bronze_id
                           JOIN public.data_quality_runs q ON q.run_id = s.quality_run_id
                           WHERE s.quality_run_id = %s
                           ON CONFLICT (event_id) DO NOTHING""",
                        (run_id,),
                    )
                    cursor.execute(
                        """INSERT INTO public.market_price_outbox
                           (bronze_id, topic, message_key, payload)
                           SELECT b.id, %s, convert_to(s.symbol, 'UTF8'), b.payload
                           FROM public.market_price_silver s
                           JOIN public.market_price_bronze b ON b.id = s.bronze_id
                           WHERE s.quality_run_id = %s
                           ON CONFLICT (bronze_id) DO NOTHING""",
                        (VALIDATED_TOPIC, run_id),
                    )
                return result

    def unsent_count(self) -> int:
        with closing(self.connect()) as connection, connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM public.market_price_outbox WHERE sent_at IS NULL")
                return cursor.fetchone()[0]

    def publish_outbox(self, producer, *, limit: int = 100) -> int:
        """At-least-once publication; a crash may repeat an event but never lose it."""
        sent = 0
        with closing(self.connect()) as connection, connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT id, topic, message_key, payload
                       FROM public.market_price_outbox
                       WHERE sent_at IS NULL ORDER BY id LIMIT %s
                       FOR UPDATE SKIP LOCKED""",
                    (limit,),
                )
                for outbox_id, topic, key, payload in cursor.fetchall():
                    producer.send(topic, key=bytes(key), value=bytes(payload)).get(timeout=15)
                    cursor.execute(
                        "UPDATE public.market_price_outbox SET sent_at = CURRENT_TIMESTAMP WHERE id = %s",
                        (outbox_id,),
                    )
                    sent += 1
        return sent
