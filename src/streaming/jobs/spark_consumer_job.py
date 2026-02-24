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

POSTGRES_URL = os.getenv("POSTGRES_URL", "jdbc:postgresql://postgres:5432/stockdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "Huyquan1607")
POSTGRES_TABLE = os.getenv("POSTGRES_TABLE", "stock_prices")


def create_spark_session(app_name: str = "KafkaSparkConsumer") -> SparkSession:
    """Create a Spark session with Kafka and PostgreSQL support."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.postgresql:postgresql:42.6.0")
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
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )


def write_to_postgres(batch_df: DataFrame, batch_id: int) -> None:
    """
    Write a batch of data to PostgreSQL.
    
    Args:
        batch_df: Spark DataFrame batch
        batch_id: Batch identifier
    """
    if batch_df.count() == 0:
        return
    
    try:
        (
            batch_df.write
            .format("jdbc")
            .option("url", POSTGRES_URL)
            .option("dbtable", POSTGRES_TABLE)
            .option("user", POSTGRES_USER)
            .option("password", POSTGRES_PASSWORD)
            .option("driver", "org.postgresql.Driver")
            .mode("append")
            .save()
        )
        print(f"✅ Batch {batch_id}: Wrote {batch_df.count()} records to PostgreSQL")
    except Exception as e:
        print(f"❌ Batch {batch_id}: Failed to write to PostgreSQL - {e}")


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