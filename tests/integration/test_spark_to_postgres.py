from pyspark.sql import SparkSession
from streaming.jobs.spark_writer_job import write_to_postgres


def test_write_to_postgres():
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("pg-test")
        .getOrCreate()
    )

    data = [{
        "ticker": "AAPL",
        "timestamp": 1700000000,
        "open": 100.0,
        "close": 102.0,
        "volume": 1000000
    }]

    df = spark.createDataFrame(data)

    # giả lập foreachBatch
    write_to_postgres(df, batch_id=1)

    spark.stop()