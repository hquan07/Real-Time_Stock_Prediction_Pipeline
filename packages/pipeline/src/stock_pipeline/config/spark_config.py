import os

SPARK_APP_NAME = os.getenv("SPARK_APP_NAME", "StockSparkApp")

SPARK_MASTER = os.getenv("SPARK_MASTER", "spark://spark-master:7077")

# Kafka integration
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "broker:29092")

# PostgreSQL sink
POSTGRES_URL = os.getenv("POSTGRES_URL", "jdbc:postgresql://postgres:5432/stockdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

# Spark configs (same as spark-defaults.conf)
SPARK_CONFIG = {
    "spark.sql.shuffle.partitions": "4",
    "spark.streaming.stopGracefullyOnShutdown": "true",
    "spark.executor.memory": "2g",
    "spark.driver.memory": "2g",
    "spark.jars.packages": ",".join([
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
        "org.postgresql:postgresql:42.6.0",
        "org.apache.spark:spark-avro_2.12:3.5.0"
    ])
}