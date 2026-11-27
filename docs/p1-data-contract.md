# P1 market-price data contract and research operating policy

Updated: 2026-11-27. Branch: `upr/p1-data-contract`.

## Source decision and limits

The owner selected **EOD research with YFinance** for P1. The default universe
is the ten US equity tickers in `get_ticker_list()`; `TICKERS` or the manual CLI
can narrow/replace it. Only `1d` XNYS bars are accepted by the new contract.
The dashboard must say research only. Neither this source choice nor an API key
approves a realtime or commercial pilot. The yfinance maintainer describes
the tool as intended for research/education and directs users to Yahoo's data
terms: <https://github.com/ranaroussi/yfinance#legal-stuff>. A later pilot
requires a separate provider, permitted use, cost/quota, universe and SLA
decision.

`event_time_utc` is the timestamp supplied with the daily bar, often the
session label at midnight New York time. It is **not** proof that the completed
bar was available then. `ingested_at_utc` is when this producer fetched it;
Gold `as_of_time` is the later of worker receipt and quality-run approval.
Model features must use the later availability time and never infer historical availability from the
session label alone.

## Research targets and data policy

| Measure | P1 research policy |
| --- | --- |
| Expected completed bar | Current XNYS session after market close plus 3 hours; otherwise previous session |
| Freshness | Latest source session must be at least the expected completed session |
| Universe completeness | `publish_yfinance` requires one bar for every configured symbol before sending any record |
| Per-symbol completeness | Missing XNYS sessions fail the quality run; historical backfill is required |
| Duplicate logical bars | 0 in Silver and Gold; key is `(source, symbol, event_time_utc, interval)` |
| Late arrival | Up to 7 days in normal operation; explicit `--historical` is for controlled replay |
| Kafka retention | Raw and validated: 7 days; DLQ: 30 days |
| Recovery targets | Research targets: RPO 24 hours, RTO 4 hours; not certified until a later restore/replay drill |

These are engineering targets for a local research stack, not a measured
external SLA. Missing data, unsupported interval, invalid OHLCV, bad schema,
future timestamps and stale sessions fail closed. The publisher does not
replace missing bars with fabricated values. A failed run is not promoted to
Gold. The scheduled training and research-signal adapters now read only
YFinance Gold rows linked to passed quality runs; insufficient backfill blocks
their output. Model evaluation/promotion still needs the P3 gate before pilot.

## Event and storage layout

- `market_price_v1.avsc` is a price-only Avro record with deterministic
  `event_id`, UTC event/ingestion timestamps, OHLCV, source, interval, timezone,
  schema version and flags. Corporate actions/fundamentals remain outside it.
- The Kafka key is `symbol`. `market_price_raw`, `market_price_validated`, and
  `market_price_dlq` each have three partitions. Schema Registry subject
  `market_price_raw-value` and `market_price_validated-value` are configured
  `BACKWARD`; messages use the raw subject's schema ID
  in Confluent framing. The old `stock_raw_data` topic remains a separate P0
  compatibility path.
- Bronze preserves the original bytes and Kafka coordinate. Silver stores the
  first valid logical bar and points to Bronze. A repeated file adds Bronze
  evidence, while Silver and Gold counts stay fixed. Conflicting corrections
  are quarantined pending an explicit versioning policy.
- `data_quality_runs` and `data_quality_results` persist checks and run IDs.
  Invalid records retain Bronze bytes, a quarantine row and a DLQ publication.
  Gold and validated-topic publication happen only after the entire run passes.
  The transactional outbox survives a worker crash; publication is at least
  once, so consumers must deduplicate by `event_id`.
- Gold carries the accepted bar, `bar_version=1`, `as_of_time`, Bronze ID and
  quality run ID. The API uses YFinance Gold first, with the legacy table as a
  research fallback while historical backfill is incomplete.

## Local operation

1. Keep the pre-P1 dump from `preflight/backup-inventory.md`. Apply migration
   `0003` with `python scripts/migrate.py up` in an environment that can reach
   the project PostgreSQL service. The migration runner checks checksums.
2. Start only P1 services with
   `docker compose --env-file .env --profile p1 up -d schema-registry market-price-topics market-price-worker`.
3. After the expected EOD session is available, publish with
   `docker compose --env-file .env --profile p1 run --rm --no-deps market-price-worker python -m stock_pipeline.ingestion.market_price.publish_yfinance`.
   Use `--tickers AAPL,MSFT` for a deliberately narrowed research run.
4. To replay an audited CSV, mount it read-only and run
   `python -m stock_pipeline.ingestion.market_price.replay /app/replay.csv --source yfinance --historical`
   inside a one-off `market-price-worker` container. CSV columns are `symbol`,
   `event_time_utc`, `open`, `high`, `low`, `close`, `volume`; timestamp must be
   ISO 8601 with timezone or UTC milliseconds. Do not label synthetic fixtures
   as `yfinance`.
5. Inspect `data_quality_runs` before using Gold. Trace a bar with
   `market_price_gold_bars.event_id → market_price_silver.bronze_id →
   market_price_bronze.id`; failed checks are in `data_quality_results` and
   `market_price_quarantine`. Check outbox unsent count if Kafka publication
   lags. Reprocess a failed file as a new run ID after fixing the source data.

The first development validation used source `fixture`, not YFinance. Two valid
bars passed raw → validated → Silver/Gold; replaying the same file added two
Bronze records and zero logical bars. A negative-price row made its run fail,
produced one quarantine/DLQ record and did not create Gold. The CI storage
drill also checks missing bars delivered out of order. A second fixture covered
one EOD session for all ten default symbols: the first run passed with nine
new bars and one existing duplicate; replaying the same ten rows yielded ten
duplicates and no increase in Silver/Gold counts. Fixture rows are isolated by
`source='fixture'` in the local development database and are excluded from the
dashboard's YFinance Gold query.

A manual YFinance AAPL EOD publish also passed on 2026-11-27. Its Gold row
traces back to the 2026-11-25 XNYS session, with separate retrieval and
availability timestamps. The dashboard API returned `source=yfinance`,
`data_tier=gold`, `research_only=true` and `stale=false` for the latest
completed session. The reverse proxy returned HTTP 200 for the page and stock
endpoint. One bar is insufficient for long-window indicators; those fields
are JSON `null`, not invented numbers.

## P0 checksum reconciliation on the local development database

Before applying `0003`, the live `schema_migrations` table held older checksums
for `0001` and `0002`. The runner correctly refused to continue. A fresh
database built from the current P0 files and a schema-only dump of the live
database were compared after removing only `pg_dump`'s random `\\restrict`
markers; the public schemas matched byte for byte. Under the migration
advisory lock, the two recorded checksums were updated only when their exact
old values matched, then `0003` applied. No baseline migration file was edited
and no application data was deleted. Other environments must perform their
own schema comparison before any checksum reconciliation.

| Migration | Recorded old checksum | Current file checksum |
| --- | --- | --- |
| `0001_baseline.sql` | `f8c6fa96ae7811154313580c97caa0b7df09091764628520552f16bb5b790b09` | `ed5463c1e039fc70c23c3abbf71fb65cd6eadbddd2bd9069bc71d469ec91be1c` |
| `0002_p0_event_contract.sql` | `045a3eb10bd28c51fc08932b4351f42e3702f00f18c998fabc45ad0b9410ceba` | `6ffcd94ce87c175d5553e015b6fffe07f4b4468abb1c4706ec5f421497bd122f` |

## Remaining P1 acceptance work

- Run the full CI branch gate. The ten-symbol one-session replay proves
  idempotency mechanics, not production throughput or measured SLA capacity.
- Decide whether a pilot provider/license and measured RPO/RTO are required
  before changing the product label or opening a pilot. They are outside the
  owner's EOD research choice for this phase.
