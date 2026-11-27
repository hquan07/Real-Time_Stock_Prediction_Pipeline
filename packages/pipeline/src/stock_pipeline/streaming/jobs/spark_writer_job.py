from pyspark.sql import SparkSession, DataFrame
from stock_pipeline.config.settings import DatabaseSettings


DB_SETTINGS = DatabaseSettings.from_env()
POSTGRES_URL = DB_SETTINGS.jdbc_url
POSTGRES_TABLE = "stock_prices"
POSTGRES_USER = DB_SETTINGS.user
POSTGRES_PASSWORD = DB_SETTINGS.password


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
