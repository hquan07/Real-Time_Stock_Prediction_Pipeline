# Streaming schema ownership

The canonical Kafka contract is
`packages/pipeline/src/stock_pipeline/ingestion/schema/stock_schema.avsc`. PostgreSQL schema changes are
owned by `infrastructure/migrations/`. Streaming jobs must not maintain copies
of either contract.
