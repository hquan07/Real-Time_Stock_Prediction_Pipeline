#!/usr/bin/env python3
"""
Simple Spark Kafka Consumer - Reads from Kafka and prints to console
Run inside Spark container or via spark-submit
"""

import sys
sys.path.insert(0, '/opt/app')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, expr
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

# Define schema for stock data
STOCK_SCHEMA = StructType([
    StructField("ticker", StringType(), True),
    StructField("timestamp", LongType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", LongType(), True),
])

def create_spark_session():
    """Create Spark session with Kafka support"""
    return (SparkSession.builder
        .appName("StockKafkaConsumer")
        .master("spark://spark-master:7077")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint")
        .getOrCreate())

def run_streaming():
    """Run streaming job"""
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    print("=" * 60)
    print("Starting Spark Streaming from Kafka...")
    print("=" * 60)
    
    # Read from Kafka
    df = (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "kafka:29092")
        .option("subscribe", "stock_raw_data")
        .option("startingOffsets", "earliest")
        .load())
    
    # Parse the data (Avro bytes - just show raw for now)
    parsed = df.selectExpr(
        "CAST(key AS STRING) as key",
        "CAST(value AS STRING) as value",
        "topic",
        "partition",
        "offset",
        "timestamp"
    )
    
    # Write to console
    query = (parsed.writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", False)
        .trigger(processingTime="5 seconds")
        .start())
    
    print("Streaming started. Press Ctrl+C to stop.")
    query.awaitTermination(timeout=60)  # Run for 60 seconds
    
    print("Streaming completed!")
    spark.stop()

if __name__ == "__main__":
    run_streaming()
