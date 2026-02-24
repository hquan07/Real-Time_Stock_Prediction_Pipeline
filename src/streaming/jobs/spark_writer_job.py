from pyspark.sql import SparkSession, DataFrame


POSTGRES_URL = "jdbc:postgresql://postgres:5432/stockdb"
POSTGRES_TABLE = "stock_prices"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "Huyquan1607"


def create_spark_session(app_name="KafkaSparkWriter") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def write_to_postgres(batch_df: DataFrame, batch_id: int):
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


def run_writer(transformed_df: DataFrame):
    query = (
        transformed_df
        .writeStream
        .foreachBatch(write_to_postgres)
        .outputMode("append")
        .start()
    )

    query.awaitTermination()