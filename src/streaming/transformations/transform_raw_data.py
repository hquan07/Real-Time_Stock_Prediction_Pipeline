import os
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.avro.functions import from_avro

# Load Avro schema
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), '../../data_ingestion/schema/stock_schema.avsc')
with open(SCHEMA_PATH, 'r') as f:
    avro_schema_str = f.read()

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
    from pyspark.sql.functions import from_unixtime

    clean_df = (
        df
        .filter(col("ticker").isNotNull())
        .filter(col("timestamp").isNotNull())
        .withColumn("event_time", from_unixtime(col("timestamp") / 1000).cast("timestamp"))
        .withColumn("open", col("price.open"))
        .withColumn("high", col("price.high"))
        .withColumn("low", col("price.low"))
        .withColumn("close", col("price.close"))
        .withColumn("volume", col("price.volume"))
        .select("ticker", "event_time", "open", "high", "low", "close", "volume")
    )

    return clean_df