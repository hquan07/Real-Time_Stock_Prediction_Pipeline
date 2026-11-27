ALTER TABLE public.signals
    DROP CONSTRAINT IF EXISTS signals_public_direction_requires_backtest,
    DROP COLUMN IF EXISTS backtest_version,
    DROP COLUMN IF EXISTS data_as_of,
    DROP COLUMN IF EXISTS model_version,
    DROP COLUMN IF EXISTS is_research_only;

DROP TABLE IF EXISTS public.predictions;
DROP INDEX IF EXISTS public.uq_stock_prices_stream_event_id;

ALTER TABLE public.stock_prices_stream
    DROP COLUMN IF EXISTS close_lag_10,
    DROP COLUMN IF EXISTS close_lag_5,
    DROP COLUMN IF EXISTS close_lag_1,
    DROP COLUMN IF EXISTS volatility_10,
    DROP COLUMN IF EXISTS ma_10,
    DROP COLUMN IF EXISTS ma_5,
    DROP COLUMN IF EXISTS log_return,
    DROP COLUMN IF EXISTS exchange_timezone,
    DROP COLUMN IF EXISTS schema_version,
    DROP COLUMN IF EXISTS event_id;

ALTER TABLE public.stock_prices_stream DROP COLUMN event_date;
ALTER TABLE public.stock_prices_stream
    ALTER COLUMN event_time TYPE timestamp
        USING event_time AT TIME ZONE 'UTC',
    ALTER COLUMN ingest_time TYPE timestamp
        USING ingest_time AT TIME ZONE 'UTC';
ALTER TABLE public.stock_prices_stream
    ADD COLUMN event_date date GENERATED ALWAYS AS (event_time::date) STORED;
