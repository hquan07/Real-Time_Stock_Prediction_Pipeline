from pyspark.sql import SparkSession
from streaming.transformations.transform_raw_data import transform_raw_data


def test_transform_raw_data_local():
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName("spark-test")
        .getOrCreate()
    )

    data = [{
        "ticker": "AAPL",
        "timestamp": 1700000000,
        "price": {
            "open": 100.0,
            "high": 105.0,
            "low": 98.0,
            "close": 102.0,
            "volume": 1000000
        }
    }]

    df = spark.createDataFrame(data)
    result = transform_raw_data(df)

    assert "open" in result.columns
    assert result.count() == 1

    spark.stop()