from pyspark.sql import DataFrame
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, IntegerType

# Spark schema
stock_schema = StructType([
    StructField("ticker", StringType(), False),
    StructField("timestamp", LongType(), False),

    StructField("price", StructType([
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", IntegerType(), True),
    ]), True),

    StructField("dividends", DoubleType(), True),
    StructField("splits", DoubleType(), True),

    StructField("earnings", StructType([
        StructField("eps_actual", DoubleType(), True),
        StructField("eps_estimate", DoubleType(), True),
        StructField("quarter", StringType(), True),
    ]), True),

    StructField("financials", StructType([
        StructField("total_revenue", LongType(), True),
        StructField("net_income", LongType(), True),
        StructField("operating_income", LongType(), True),
    ]), True),
])

# Parse + Valid
def parse_kafka_value(df: DataFrame) -> DataFrame:
    """
    Parse Kafka value (JSON string) to structured columns
    """
    return (
        df.selectExpr("CAST(value AS STRING) as json")
          .select(from_json(col("json"), stock_schema).alias("data"))
          .select("data.*")
    )

# Clean and Transform
def transform_raw_data(df: DataFrame) -> DataFrame:
    """
    Clean + flatten raw stock data
    """

    clean_df = (
        df
        .filter(col("ticker").isNotNull())
        .filter(col("timestamp").isNotNull())
        .withColumn("open", col("price.open"))
        .withColumn("high", col("price.high"))
        .withColumn("low", col("price.low"))
        .withColumn("close", col("price.close"))
        .withColumn("volume", col("price.volume"))
        .drop("price")
    )

    return clean_df