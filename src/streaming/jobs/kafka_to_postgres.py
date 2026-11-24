#!/usr/bin/env python3
"""
Kafka to PostgreSQL Consumer
Reads stock data from Kafka and writes to PostgreSQL database
"""

import sys
import json
from io import BytesIO
from datetime import datetime
from pathlib import Path

# Ensure src is in path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kafka import KafkaConsumer
import psycopg2
from fastavro import schemaless_reader
from loguru import logger
from config.settings import DatabaseSettings, KafkaSettings

from fastavro import parse_schema

DB_SETTINGS = DatabaseSettings.from_env()
KAFKA_SETTINGS = KafkaSettings.from_env()
KAFKA_BOOTSTRAP = KAFKA_SETTINGS.bootstrap_servers
KAFKA_TOPIC = KAFKA_SETTINGS.topic

# Avro schema for deserialization
with KAFKA_SETTINGS.schema_path.open("r") as f:
    STOCK_SCHEMA = parse_schema(json.load(f))




def get_db_connection():
    """Create PostgreSQL connection"""
    return psycopg2.connect(
        **DB_SETTINGS.connect_kwargs()
    )


def deserialize_avro(data: bytes) -> dict:
    """Deserialize Avro bytes to dict"""
    try:
        buffer = BytesIO(data)
        return schemaless_reader(buffer, STOCK_SCHEMA)
    except Exception as e:
        logger.warning(f"Failed to deserialize: {e}")
        return None


def write_to_db(conn, stock_data: dict):
    """Write stock data to price_history table"""
    try:
        cur = conn.cursor()
        ticker = stock_data['ticker']
        
        # Convert timestamp from milliseconds to datetime
        ts = datetime.fromtimestamp(stock_data['timestamp'] / 1000)
        
        # Ensure company exists (foreign key)
        cur.execute("""
            INSERT INTO companies (ticker, long_name)
            VALUES (%s, %s)
            ON CONFLICT (ticker) DO NOTHING
        """, (ticker, ticker))
        
        # Insert into stock_prices_stream
        insert_sql = """
            INSERT INTO stock_prices_stream 
            (ticker, event_time, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, event_time) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """
        
        price_data = stock_data.get('price') or {}
        
        cur.execute(insert_sql, (
            ticker,
            ts,
            price_data.get('open'),
            price_data.get('high'),
            price_data.get('low'),
            price_data.get('close'),
            price_data.get('volume')
        ))
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"DB write error: {e}")
        conn.rollback()
        return False


def run_consumer():
    """Main consumer loop"""
    logger.info("=" * 60)
    logger.info("Starting Kafka to PostgreSQL Consumer")
    logger.info("=" * 60)
    
    # Connect to PostgreSQL
    conn = get_db_connection()
    logger.info(
        f"✅ Connected to PostgreSQL {DB_SETTINGS.host}:"
        f"{DB_SETTINGS.port}/{DB_SETTINGS.database}"
    )
    
    # Create Kafka consumer
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset='earliest',
        consumer_timeout_ms=30000,  # 30 seconds timeout
        value_deserializer=lambda x: x  # Raw bytes
    )
    logger.info(f"✅ Connected to Kafka {KAFKA_BOOTSTRAP}")
    logger.info(f"📨 Consuming from topic: {KAFKA_TOPIC}")
    
    count = 0
    success = 0
    
    for message in consumer:
        count += 1
        
        # Deserialize Avro
        stock_data = deserialize_avro(message.value)
        
        if stock_data:
            ticker = stock_data.get('ticker', 'UNKNOWN')
            
            # Write to PostgreSQL
            if write_to_db(conn, stock_data):
                success += 1
                logger.info(f"✅ [{count}] Wrote {ticker} to PostgreSQL")
            else:
                logger.warning(f"⚠️ [{count}] Failed to write {ticker}")
        else:
            logger.warning(f"⚠️ [{count}] Failed to deserialize message")
    
    # Cleanup
    consumer.close()
    conn.close()
    
    logger.info("=" * 60)
    logger.info(f"✅ Completed! Processed {count} messages, {success} written to DB")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_consumer()
