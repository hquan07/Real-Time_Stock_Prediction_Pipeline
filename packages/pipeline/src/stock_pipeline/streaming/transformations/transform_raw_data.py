from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, timestamp_millis
from pyspark.sql.avro.functions import from_avro

# Load Avro schema
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "ingestion/schema/stock_schema.avsc"
avro_schema_str = SCHEMA_PATH.read_text()

# Parse + Valid
def parse_kafka_value(df: DataFrame) -> DataFrame:
    """
    Parse Kafka value (Avro binary) to structured columns
    """
    return (
        df.select(from_avro(col("value"), avro_schema_str).alias("data"))
          .select("data.*")
    )

# Clean and Transform
def transform_raw_data(df: DataFrame) -> DataFrame:
    """
    Clean + flatten raw stock data and map to stock_prices_stream schema
    """
    clean_df = (
        df
        .filter(col("ticker").isNotNull())
        .filter(col("timestamp").isNotNull())
        .filter(col("event_id").isNotNull())
        .withColumn("event_time", timestamp_millis(col("timestamp")))
        .withColumn("open", col("price.open"))
        .withColumn("high", col("price.high"))
        .withColumn("low", col("price.low"))
        .withColumn("close", col("price.close"))
        .withColumn("volume", col("price.volume"))
        .select(
            "event_id",
            "schema_version",
            "ticker",
            "event_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "interval",
            "source",
            "exchange_timezone",
        )
    )

    return clean_df
