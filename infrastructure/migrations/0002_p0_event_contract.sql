-- P0 event contract: UTC timestamps, deterministic deduplication and research-only signals.

ALTER TABLE public.stock_prices_stream DROP COLUMN event_date;

ALTER TABLE public.stock_prices_stream
    ALTER COLUMN event_time TYPE timestamptz
        USING event_time AT TIME ZONE 'UTC',
    ALTER COLUMN ingest_time TYPE timestamptz
        USING ingest_time AT TIME ZONE 'UTC';

ALTER TABLE public.stock_prices_stream
    ADD COLUMN event_date date GENERATED ALWAYS AS
        ((event_time AT TIME ZONE 'UTC')::date) STORED;

ALTER TABLE public.stock_prices_stream
    ADD COLUMN IF NOT EXISTS event_id varchar(64),
    ADD COLUMN IF NOT EXISTS schema_version integer,
    ADD COLUMN IF NOT EXISTS exchange_timezone varchar(64),
    ADD COLUMN IF NOT EXISTS log_return double precision,
    ADD COLUMN IF NOT EXISTS ma_5 double precision,
    ADD COLUMN IF NOT EXISTS ma_10 double precision,
    ADD COLUMN IF NOT EXISTS volatility_10 double precision,
    ADD COLUMN IF NOT EXISTS close_lag_1 double precision,
    ADD COLUMN IF NOT EXISTS close_lag_5 double precision,
    ADD COLUMN IF NOT EXISTS close_lag_10 double precision;

UPDATE public.stock_prices_stream
SET source = COALESCE(source, 'legacy'),
    interval = COALESCE(interval, '1d'),
    schema_version = COALESCE(schema_version, 1),
    exchange_timezone = COALESCE(exchange_timezone, 'America/New_York'),
    event_id = COALESCE(
        event_id,
        md5(ticker || '|' || extract(epoch FROM event_time)::text || '|' || COALESCE(source, 'legacy'))
        || md5(ticker || '|' || extract(epoch FROM event_time)::text || '|legacy')
    );

ALTER TABLE public.stock_prices_stream
    ALTER COLUMN event_id SET NOT NULL,
    ALTER COLUMN schema_version SET DEFAULT 2,
    ALTER COLUMN schema_version SET NOT NULL,
    ALTER COLUMN source SET DEFAULT 'yfinance',
    ALTER COLUMN source SET NOT NULL,
    ALTER COLUMN interval SET DEFAULT '1d',
    ALTER COLUMN interval SET NOT NULL,
    ALTER COLUMN exchange_timezone SET DEFAULT 'America/New_York',
    ALTER COLUMN exchange_timezone SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_stock_prices_stream_event_id
    ON public.stock_prices_stream (event_id);

CREATE TABLE IF NOT EXISTS public.predictions (
    id bigserial PRIMARY KEY,
    ticker varchar(20) NOT NULL,
    prediction_date date NOT NULL,
    target_date date NOT NULL,
    model_name varchar(100) NOT NULL,
    model_version varchar(100),
    predicted_close numeric,
    predicted_return numeric,
    confidence numeric,
    uncertainty_validated boolean NOT NULL DEFAULT false,
    features_used jsonb,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, prediction_date, target_date, model_name)
);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON public.predictions (ticker);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON public.predictions (prediction_date);

ALTER TABLE public.signals
    ADD COLUMN IF NOT EXISTS is_research_only boolean NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS model_version varchar(100),
    ADD COLUMN IF NOT EXISTS data_as_of timestamptz,
    ADD COLUMN IF NOT EXISTS backtest_version varchar(100);

ALTER TABLE public.signals
    DROP CONSTRAINT IF EXISTS signals_public_direction_requires_backtest;
ALTER TABLE public.signals
    ADD CONSTRAINT signals_public_direction_requires_backtest
    CHECK (is_research_only OR (signal_type = 'HOLD' AND backtest_version IS NOT NULL));
