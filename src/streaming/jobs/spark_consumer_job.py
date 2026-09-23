"""
Spark Streaming Consumer Job.
Reads from Kafka, transforms data, and writes to PostgreSQL.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pyspark.sql import SparkSession, DataFrame

from ..transformations.transform_raw_data import (
    parse_kafka_value,
    transform_raw_data
)


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "stock_raw_data")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "stockdb")
POSTGRES_URL = os.getenv("POSTGRES_URL", f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_TABLE = os.getenv("POSTGRES_TABLE", "stock_prices_stream")


def create_spark_session(app_name: str = "KafkaSparkConsumer") -> SparkSession:
    """Create a Spark session with Kafka and PostgreSQL support."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.postgresql:postgresql:42.6.0,"
                "org.apache.spark:spark-avro_2.12:3.5.0")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint")
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
        from sqlalchemy import create_engine
        engine = create_engine(POSTGRES_URL.replace("jdbc:postgresql", "postgresql+psycopg2"))
        
        tickers = tuple(batch_pdf["ticker"].unique())
        if len(tickers) == 1:
            tickers_str = f"('{tickers[0]}')"
        else:
            tickers_str = str(tickers)
            
        history_query = f"""
            SELECT * FROM (
                SELECT ticker, event_time as date, open, high, low, close, volume,
                ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY event_time DESC) as rn
                FROM {POSTGRES_TABLE}
                WHERE ticker IN {tickers_str}
            ) tmp WHERE rn <= 15
        """
        history_pdf = pd.read_sql(history_query, engine)
        history_pdf = history_pdf.drop(columns=["rn"])
        
        # Format batch to match history
        batch_pdf = batch_pdf.rename(columns={"event_time": "date"})
        
        # Combine
        combined_pdf = pd.concat([history_pdf, batch_pdf]).drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
        
        # 3. Apply Feature Engineering
        from machine_learning.feature_engineering.features import build_features
        features_pdf = pd.DataFrame()
        for ticker, group in combined_pdf.groupby("ticker"):
            feat_group = build_features(group)
            features_pdf = pd.concat([features_pdf, feat_group])
            
        # Filter only new records (from batch)
        batch_dates = batch_pdf["date"].tolist()
        features_pdf = features_pdf[features_pdf["date"].isin(batch_dates)]
        
        if features_pdf.empty:
            print(f"⚠️ Batch {batch_id}: No valid features generated (not enough history?)")
            # Write raw data anyway
            features_pdf = batch_pdf
            
        # 4. Write back to Postgres
        # We rename date back to event_time if needed, but since we are replacing the logic, let's keep it simple
        features_pdf = features_pdf.rename(columns={"date": "event_time"})
        
        # Convert back to Spark DF to write efficiently, or just use pandas to_sql
        features_pdf.to_sql(POSTGRES_TABLE, engine, if_exists='append', index=False)
        print(f"✅ Batch {batch_id}: Wrote {len(features_pdf)} records with features to PostgreSQL")
        
    except Exception as e:
        print(f"❌ Batch {batch_id}: Failed to process/write to PostgreSQL - {e}")



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
    checkpoint_location: str = "/tmp/spark-checkpoint"
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