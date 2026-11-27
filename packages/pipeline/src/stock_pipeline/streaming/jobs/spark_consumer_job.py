"""
Spark Streaming Consumer Job.
Reads from Kafka, transforms data, and writes to PostgreSQL.
"""

from __future__ import annotations

import os
import sys
import re

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pyspark.sql import SparkSession, DataFrame
from stock_pipeline.config.settings import DatabaseSettings, KafkaSettings

from stock_pipeline.streaming.transformations.transform_raw_data import (
    parse_kafka_value,
    transform_raw_data
)


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
KAFKA_SETTINGS = KafkaSettings.from_env()
DB_SETTINGS = DatabaseSettings.from_env()
KAFKA_BOOTSTRAP_SERVERS = KAFKA_SETTINGS.bootstrap_servers
KAFKA_TOPIC = KAFKA_SETTINGS.topic
POSTGRES_URL = DB_SETTINGS.jdbc_url
POSTGRES_TABLE = os.getenv("POSTGRES_TABLE", "stock_prices_stream")
CHECKPOINT_LOCATION = os.getenv(
    "SPARK_CHECKPOINT_LOCATION", "/opt/spark/checkpoints/stock-consumer"
)
if not re.fullmatch(r"[a-z_][a-z0-9_]*", POSTGRES_TABLE):
    raise ValueError("POSTGRES_TABLE must be a simple SQL identifier")


def create_spark_session(app_name: str = "KafkaSparkConsumer") -> SparkSession:
    """Create a Spark session with Kafka and PostgreSQL support."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,"
                "org.postgresql:postgresql:42.6.0,"
                "org.apache.spark:spark-avro_2.12:3.5.3")
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_LOCATION)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def read_from_kafka(spark: SparkSession) -> DataFrame:
    """Read streaming data from Kafka topic."""
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )


def write_to_postgres(batch_df: DataFrame, batch_id: int) -> None:
    if batch_df.count() == 0:
        return
    
    # 1. Convert micro-batch to Pandas
    batch_pdf = batch_df.toPandas()
    
    # 2. Get last 15 records from Postgres to calculate rolling features
    import pandas as pd
    try:
        from sqlalchemy import bindparam, create_engine, text
        engine = create_engine(DB_SETTINGS.sqlalchemy_url)
        
        tickers = tuple(batch_pdf["ticker"].unique())
        history_query = text(f"""
            SELECT * FROM (
                SELECT event_id, schema_version, ticker, event_time as date,
                       open, high, low, close, volume, interval, source,
                       exchange_timezone,
                ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY event_time DESC) as rn
                FROM {POSTGRES_TABLE}
                WHERE ticker IN :tickers
            ) tmp WHERE rn <= 15
        """).bindparams(bindparam("tickers", expanding=True))
        history_pdf = pd.read_sql(history_query, engine, params={"tickers": tickers})
        history_pdf = history_pdf.drop(columns=["rn"])
        
        # Format batch to match history
        batch_pdf = batch_pdf.rename(columns={"event_time": "date"})
        
        # Combine
        combined_pdf = pd.concat([history_pdf, batch_pdf]).drop_duplicates(
            subset=["event_id"], keep="last"
        ).sort_values(["ticker", "date"])
        
        # 3. Apply Feature Engineering
        from stock_pipeline.ml.feature_engineering.features import build_features
        features_pdf = pd.DataFrame()
        for ticker, group in combined_pdf.groupby("ticker"):
            feat_group = build_features(group, is_training=False)
            features_pdf = pd.concat([features_pdf, feat_group])
            
        # Filter only new records (from batch)
        batch_dates = batch_pdf["date"].tolist()
        features_pdf = features_pdf[features_pdf["date"].isin(batch_dates)]
        
        if features_pdf.empty:
            print(f"⚠️ Batch {batch_id}: no feature-complete rows; writing validated raw rows")
            features_pdf = batch_pdf.copy()
            
        features_pdf = features_pdf.rename(columns={"date": "event_time"})
        upsert_feature_rows(engine, features_pdf.to_dict(orient="records"))
        print(f"✅ Batch {batch_id}: Wrote {len(features_pdf)} records with features to PostgreSQL")
        
    except Exception as e:
        print(f"❌ Batch {batch_id}: Failed to process/write to PostgreSQL - {e}")
        raise


def upsert_feature_rows(engine, rows: list[dict]) -> None:
    """Idempotently persist one micro-batch using the canonical event_id."""
    from sqlalchemy import text

    columns = [
        "event_id", "schema_version", "ticker", "event_time", "open", "high",
        "low", "close", "volume", "interval", "source", "exchange_timezone",
        "log_return", "ma_5", "ma_10", "volatility_10", "close_lag_1",
        "close_lag_5", "close_lag_10",
    ]
    statement = text(f"""
        INSERT INTO {POSTGRES_TABLE} ({', '.join(columns)})
        VALUES ({', '.join(':' + column for column in columns)})
        ON CONFLICT (event_id) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            log_return = EXCLUDED.log_return,
            ma_5 = EXCLUDED.ma_5,
            ma_10 = EXCLUDED.ma_10,
            volatility_10 = EXCLUDED.volatility_10,
            close_lag_1 = EXCLUDED.close_lag_1,
            close_lag_5 = EXCLUDED.close_lag_5,
            close_lag_10 = EXCLUDED.close_lag_10,
            ingest_time = CURRENT_TIMESTAMP
    """)
    normalized_rows = [
        {column: row.get(column) for column in columns}
        for row in rows
    ]
    with engine.begin() as connection:
        connection.execute(statement, normalized_rows)



def write_to_console(batch_df: DataFrame, batch_id: int) -> None:
    """
    Write a batch of data to console (for debugging).
    
    Args:
        batch_df: Spark DataFrame batch
        batch_id: Batch identifier
    """
    print(f"--- Batch {batch_id} ---")
    batch_df.show(truncate=False)


def run_consumer(
    output_mode: str = "postgres",
    checkpoint_location: str = CHECKPOINT_LOCATION,
) -> None:
    """
    Run the Kafka to PostgreSQL streaming consumer.
    
    Args:
        output_mode: "postgres" to write to database, "console" for debugging
        checkpoint_location: Path for checkpoint files
    """
    spark = create_spark_session()

    # Read from Kafka
    raw_df = read_from_kafka(spark)

    # Parse and transform
    parsed_df = parse_kafka_value(raw_df)
    transformed_df = transform_raw_data(parsed_df)

    # Select output sink
    if output_mode == "postgres":
        write_fn = write_to_postgres
    else:
        write_fn = write_to_console

    # Start streaming query
    query = (
        transformed_df
        .writeStream
        .foreachBatch(write_fn)
        .outputMode("append")
        .option("checkpointLocation", checkpoint_location)
        .start()
    )

    print(f"🚀 Spark consumer started, writing to {output_mode}")
    query.awaitTermination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spark Streaming Consumer")
    parser.add_argument(
        "--output",
        choices=["postgres", "console"],
        default="postgres",
        help="Output destination"
    )
    args = parser.parse_args()

    run_consumer(output_mode=args.output)
