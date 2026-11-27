-- P1: separate versioned market-price data from the legacy mixed stock event.
-- Bronze is append-only by application policy; one Kafka offset may be retried.

CREATE TABLE public.market_price_bronze (
    id bigserial PRIMARY KEY,
    kafka_topic varchar(128) NOT NULL,
    kafka_partition integer NOT NULL,
    kafka_offset bigint NOT NULL,
    payload bytea NOT NULL,
    payload_sha256 char(64) NOT NULL,
    schema_id integer,
    run_id varchar(128) NOT NULL,
    received_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (kafka_topic, kafka_partition, kafka_offset)
);

CREATE TABLE public.data_quality_runs (
    run_id varchar(128) PRIMARY KEY,
    expected_records integer NOT NULL CHECK (expected_records > 0),
    processed_records integer NOT NULL DEFAULT 0 CHECK (processed_records >= 0),
    accepted_records integer NOT NULL DEFAULT 0 CHECK (accepted_records >= 0),
    rejected_records integer NOT NULL DEFAULT 0 CHECK (rejected_records >= 0),
    duplicate_records integer NOT NULL DEFAULT 0 CHECK (duplicate_records >= 0),
    status varchar(16) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'passed', 'failed')),
    started_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at timestamptz,
    CHECK (processed_records <= expected_records)
);

CREATE TABLE public.data_quality_results (
    id bigserial PRIMARY KEY,
    run_id varchar(128) NOT NULL REFERENCES public.data_quality_runs(run_id),
    bronze_id bigint NOT NULL REFERENCES public.market_price_bronze(id),
    event_id char(64),
    check_code varchar(64) NOT NULL,
    passed boolean NOT NULL,
    checked_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (bronze_id, check_code)
);

CREATE TABLE public.market_price_quarantine (
    bronze_id bigint PRIMARY KEY REFERENCES public.market_price_bronze(id),
    reason_codes text[] NOT NULL,
    quarantined_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE public.market_price_silver (
    event_id char(64) PRIMARY KEY,
    source varchar(50) NOT NULL,
    symbol varchar(20) NOT NULL,
    interval varchar(10) NOT NULL,
    event_time_utc timestamptz NOT NULL,
    ingested_at_utc timestamptz NOT NULL,
    open double precision NOT NULL CHECK (open > 0),
    high double precision NOT NULL CHECK (high > 0),
    low double precision NOT NULL CHECK (low > 0),
    close double precision NOT NULL CHECK (close > 0),
    volume bigint NOT NULL CHECK (volume >= 0),
    exchange_timezone varchar(64) NOT NULL,
    schema_version integer NOT NULL,
    bronze_id bigint NOT NULL REFERENCES public.market_price_bronze(id),
    quality_run_id varchar(128) NOT NULL REFERENCES public.data_quality_runs(run_id),
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source, symbol, event_time_utc, interval)
);
CREATE INDEX idx_p1_silver_symbol_time
    ON public.market_price_silver (symbol, event_time_utc DESC);

CREATE TABLE public.market_price_gold_bars (
    event_id char(64) PRIMARY KEY REFERENCES public.market_price_silver(event_id),
    bar_version integer NOT NULL DEFAULT 1,
    source varchar(50) NOT NULL,
    symbol varchar(20) NOT NULL,
    interval varchar(10) NOT NULL,
    event_time_utc timestamptz NOT NULL,
    as_of_time timestamptz NOT NULL,
    open double precision NOT NULL,
    high double precision NOT NULL,
    low double precision NOT NULL,
    close double precision NOT NULL,
    volume bigint NOT NULL,
    bronze_id bigint NOT NULL REFERENCES public.market_price_bronze(id),
    quality_run_id varchar(128) NOT NULL REFERENCES public.data_quality_runs(run_id),
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_p1_gold_symbol_time
    ON public.market_price_gold_bars (symbol, event_time_utc DESC);

-- Transactional outbox prevents a DB commit from silently losing the next
-- Kafka publication. Retrying an outbox row may repeat a message; event_id is
-- the downstream deduplication key.
CREATE TABLE public.market_price_outbox (
    id bigserial PRIMARY KEY,
    bronze_id bigint NOT NULL UNIQUE REFERENCES public.market_price_bronze(id),
    topic varchar(128) NOT NULL,
    message_key bytea NOT NULL,
    payload bytea NOT NULL,
    sent_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);
