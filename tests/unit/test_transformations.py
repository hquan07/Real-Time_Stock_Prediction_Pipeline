from pyspark.sql import SparkSession
from streaming.transformations.transform_raw_data import transform_raw_data


def test_transform_schema():
    spark = SparkSession.builder.master("local[1]").getOrCreate()

    data = [{
        "ticker": "MSFT",
        "timestamp": 1700000000,
        "price": {
            "open": 200,
            "high": 210,
            "low": 195,
            "close": 205,
            "volume": 500000
        }
    }]

    df = spark.createDataFrame(data)
    result = transform_raw_data(df)

    assert result.select("open").first()[0] == 200
    assert result.select("volume").first()[0] == 500000

    spark.stop()