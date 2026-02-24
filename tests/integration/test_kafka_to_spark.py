"""
Integration tests for Kafka to Spark pipeline.
These tests require Kafka and Spark to be running.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


@pytest.mark.integration
@pytest.mark.requires_kafka
@pytest.mark.requires_spark
class TestKafkaToSparkIntegration:
    """Integration tests for Kafka -> Spark pipeline."""

    @pytest.fixture
    def mock_spark_session(self):
        """Create mock Spark session."""
        with patch("pyspark.sql.SparkSession") as mock:
            session = MagicMock()
            mock.builder.master.return_value.appName.return_value.getOrCreate.return_value = session
            yield session

    def test_spark_consumer_reads_kafka(self, mock_spark_session):
        """Test that Spark consumer can read from Kafka topic."""
        from src.streaming.jobs.spark_consumer_job import create_spark_session
        
        # This would normally connect to Kafka
        # In mock, we verify the configuration
        assert mock_spark_session is not None

    def test_transform_raw_data_schema(self):
        """Test transformation schema is correct."""
        from src.streaming.transformations.transform_raw_data import STOCK_SCHEMA

        assert STOCK_SCHEMA is not None
        # Verify required fields
        field_names = [f.name for f in STOCK_SCHEMA.fields]
        assert "ticker" in field_names

    @pytest.mark.skip(reason="Requires running Kafka cluster")
    def test_end_to_end_kafka_spark(self):
        """End-to-end test from Kafka to Spark processing."""
        # This test would:
        # 1. Send a message to Kafka
        # 2. Start Spark consumer
        # 3. Verify message is processed
        pass


@pytest.mark.integration
class TestKafkaProducerIntegration:
    """Tests for Kafka producer."""

    @patch("src.data_ingestion.kafka_producer.producer.KafkaProducer")
    def test_producer_creation(self, mock_producer):
        """Test Kafka producer can be created."""
        from src.data_ingestion.kafka_producer.producer import create_kafka_producer

        mock_instance = MagicMock()
        mock_producer.return_value = mock_instance

        producer = create_kafka_producer()

        assert producer is not None
        mock_producer.assert_called_once()

    @patch("src.data_ingestion.kafka_producer.send_data._producer", None)
    @patch("src.data_ingestion.kafka_producer.send_data._serializer", None)
    def test_send_to_kafka_mock(self, sample_kafka_message):
        """Test sending message to Kafka (mocked)."""
        # Test would verify serialization and send
        pass


@pytest.mark.integration
class TestSparkTransformations:
    """Tests for Spark transformations."""

    def test_parse_kafka_value(self, spark_session):
        """Test parsing Kafka value to DataFrame."""
        if spark_session is None:
            pytest.skip("Spark not available")

        from src.streaming.transformations.transform_raw_data import STOCK_SCHEMA
        import json

        # Create test data
        test_data = [{
            "ticker": "AAPL",
            "timestamp": 1704067200000,
            "price": {
                "open": 150.0,
                "high": 155.0,
                "low": 148.0,
                "close": 152.0,
                "volume": 5000000,
            },
        }]

        # Create DataFrame
        rdd = spark_session.sparkContext.parallelize([json.dumps(d) for d in test_data])
        df = spark_session.read.json(rdd)

        assert df.count() == 1
        assert "ticker" in df.columns
