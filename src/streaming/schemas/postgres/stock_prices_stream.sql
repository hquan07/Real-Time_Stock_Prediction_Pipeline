-- ============================================
-- RTSPP-29: Streaming stock prices (raw layer)
-- ============================================

CREATE TABLE IF NOT EXISTS stock_prices_stream (
    id BIGSERIAL PRIMARY KEY,

    ticker VARCHAR(20) NOT NULL,

    event_time TIMESTAMP NOT NULL,
    ingest_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,

    interval VARCHAR(10),
    source VARCHAR(50),

    -- for idempotency / dedup (optional)
    event_date DATE GENERATED ALWAYS AS (event_time::date) STORED
);

-- Minimal index for downstream batch merge
CREATE INDEX IF NOT EXISTS idx_stock_stream_ticker_time
ON stock_prices_stream (ticker, event_time);