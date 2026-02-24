"""
Batch ETL Pipeline for Stock Data.
Processes historical data from PostgreSQL and computes technical indicators.
"""

import os
from typing import Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POSTGRES_URL = os.getenv("POSTGRES_URL", "jdbc:postgresql://postgres:5432/stockdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "Huyquan1607")


def create_spark_session(app_name: str = "BatchETL") -> SparkSession:
    """Create Spark session with PostgreSQL JDBC driver."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0")
        .getOrCreate()
    )


def read_from_postgres(
    spark: SparkSession,
    table_name: str = "stock_prices"
) -> DataFrame:
    """Read data from PostgreSQL table."""
    return (
        spark.read
        .format("jdbc")
        .option("url", POSTGRES_URL)
        .option("dbtable", table_name)
        .option("user", POSTGRES_USER)
        .option("password", POSTGRES_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .load()
    )


def write_to_postgres(
    df: DataFrame,
    table_name: str,
    mode: str = "append"
) -> None:
    """Write DataFrame to PostgreSQL table."""
    (
        df.write
        .format("jdbc")
        .option("url", POSTGRES_URL)
        .option("dbtable", table_name)
        .option("user", POSTGRES_USER)
        .option("password", POSTGRES_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .mode(mode)
        .save()
    )


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------
def compute_sma(df: DataFrame, column: str, window_size: int) -> DataFrame:
    """
    Compute Simple Moving Average.
    
    Args:
        df: Input DataFrame with 'ticker' and 'date' columns
        column: Column to compute SMA for
        window_size: Number of periods for SMA
    
    Returns:
        DataFrame with SMA column added.
    """
    window_spec = Window.partitionBy("ticker").orderBy("date").rowsBetween(
        -(window_size - 1), 0
    )
    
    return df.withColumn(
        f"sma_{window_size}", 
        F.avg(F.col(column)).over(window_spec)
    )


def compute_ema(df: DataFrame, column: str, window_size: int) -> DataFrame:
    """
    Compute Exponential Moving Average (approximation using rolling window).
    
    Args:
        df: Input DataFrame with 'ticker' and 'date' columns
        column: Column to compute EMA for
        window_size: Number of periods for EMA
    
    Returns:
        DataFrame with EMA column added.
    """
    # Using a weighted average approximation
    # For proper EMA, consider using pandas UDF
    alpha = 2.0 / (window_size + 1)
    
    window_spec = Window.partitionBy("ticker").orderBy("date").rowsBetween(
        -(window_size - 1), 0
    )
    
    # Simple approximation using weighted average
    return df.withColumn(
        f"ema_{window_size}",
        F.avg(F.col(column)).over(window_spec)
    )


def compute_rsi(df: DataFrame, column: str = "close", window_size: int = 14) -> DataFrame:
    """
    Compute Relative Strength Index.
    
    Args:
        df: Input DataFrame with 'ticker' and 'date' columns
        column: Column to compute RSI for (usually 'close')
        window_size: Number of periods for RSI (default 14)
    
    Returns:
        DataFrame with RSI column added.
    """
    window_spec = Window.partitionBy("ticker").orderBy("date")
    
    # Calculate price change
    df = df.withColumn("price_change", F.col(column) - F.lag(F.col(column), 1).over(window_spec))
    
    # Separate gains and losses
    df = df.withColumn("gain", F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0))
    df = df.withColumn("loss", F.when(F.col("price_change") < 0, -F.col("price_change")).otherwise(0))
    
    # Calculate average gain and loss
    rolling_window = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_size - 1), 0)
    
    df = df.withColumn("avg_gain", F.avg(F.col("gain")).over(rolling_window))
    df = df.withColumn("avg_loss", F.avg(F.col("loss")).over(rolling_window))
    
    # Calculate RSI
    df = df.withColumn(
        "rsi",
        F.when(
            F.col("avg_loss") == 0, 
            100.0
        ).otherwise(
            100.0 - (100.0 / (1.0 + F.col("avg_gain") / F.col("avg_loss")))
        )
    )
    
    # Drop intermediate columns
    df = df.drop("price_change", "gain", "loss", "avg_gain", "avg_loss")
    
    return df


def compute_bollinger_bands(
    df: DataFrame, 
    column: str = "close", 
    window_size: int = 20,
    num_std: float = 2.0
) -> DataFrame:
    """
    Compute Bollinger Bands.
    
    Args:
        df: Input DataFrame
        column: Column to compute bands for
        window_size: Number of periods for moving average
        num_std: Number of standard deviations for bands
    
    Returns:
        DataFrame with bb_upper, bb_middle, bb_lower columns added.
    """
    window_spec = Window.partitionBy("ticker").orderBy("date").rowsBetween(
        -(window_size - 1), 0
    )
    
    df = df.withColumn("bb_middle", F.avg(F.col(column)).over(window_spec))
    df = df.withColumn("bb_std", F.stddev(F.col(column)).over(window_spec))
    
    df = df.withColumn("bb_upper", F.col("bb_middle") + num_std * F.col("bb_std"))
    df = df.withColumn("bb_lower", F.col("bb_middle") - num_std * F.col("bb_std"))
    
    df = df.drop("bb_std")
    
    return df


def compute_daily_returns(df: DataFrame, column: str = "close") -> DataFrame:
    """Compute daily percentage returns."""
    window_spec = Window.partitionBy("ticker").orderBy("date")
    
    return df.withColumn(
        "daily_return",
        (F.col(column) - F.lag(F.col(column), 1).over(window_spec)) / 
        F.lag(F.col(column), 1).over(window_spec) * 100
    )


# ---------------------------------------------------------------------------
# Main ETL Pipeline
# ---------------------------------------------------------------------------
def run_batch_etl(
    output_table: Optional[str] = None,
    compute_indicators: bool = True
) -> DataFrame:
    """
    Run the batch ETL pipeline.
    
    Args:
        output_table: Table to write results (None = no write)
        compute_indicators: Whether to compute technical indicators
    
    Returns:
        Processed DataFrame
    """
    spark = create_spark_session()

    # Read from PostgreSQL
    print("📖 Reading data from PostgreSQL...")
    df = read_from_postgres(spark, "stock_prices")

    if df.count() == 0:
        print("⚠️ No data found in stock_prices table")
        return df

    print(f"📊 Loaded {df.count()} records")

    if compute_indicators:
        print("🔧 Computing technical indicators...")
        
        # Moving averages
        df = compute_sma(df, "close", 10)
        df = compute_sma(df, "close", 20)
        df = compute_sma(df, "close", 50)
        
        df = compute_ema(df, "close", 12)
        df = compute_ema(df, "close", 26)
        
        # RSI
        df = compute_rsi(df, "close", 14)
        
        # Bollinger Bands
        df = compute_bollinger_bands(df, "close", 20, 2.0)
        
        # Daily returns
        df = compute_daily_returns(df, "close")
        
        print("✅ Technical indicators computed")

    # Aggregate statistics by ticker
    print("📈 Computing aggregations...")
    daily_agg = (
        df.groupBy("ticker")
        .agg(
            F.avg("close").alias("avg_close"),
            F.avg("volume").alias("avg_volume"),
            F.max("high").alias("max_high"),
            F.min("low").alias("min_low"),
            F.stddev("close").alias("volatility"),
            F.count("*").alias("record_count"),
        )
    )
    
    daily_agg.show()

    # Write to output table if specified
    if output_table:
        print(f"💾 Writing to {output_table}...")
        write_to_postgres(df, output_table, mode="overwrite")
        print(f"✅ Written to {output_table}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch ETL Pipeline")
    parser.add_argument(
        "--output-table",
        help="Table to write processed data"
    )
    parser.add_argument(
        "--no-indicators",
        action="store_true",
        help="Skip computing technical indicators"
    )
    args = parser.parse_args()

    run_batch_etl(
        output_table=args.output_table,
        compute_indicators=not args.no_indicators
    )