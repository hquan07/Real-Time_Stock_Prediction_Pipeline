#!/usr/bin/env python3
"""
Kafka to PostgreSQL Consumer
Reads stock data from Kafka and writes to PostgreSQL database
"""

import sys
import json
from io import BytesIO
from datetime import datetime

# Ensure src is in path
sys.path.insert(0, '/home/hquan07/Real-Time_Stock_Prediction_Pipeline')

from kafka import KafkaConsumer
import psycopg2
from fastavro import schemaless_reader
from loguru import logger

# Kafka config
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "stock_raw_data"

# PostgreSQL config  
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "stockdb"
DB_USER = "postgres"
DB_PASSWORD = "Huyquan1607"

import os
from fastavro import parse_schema

# Avro schema for deserialization
SCHEMA_PATH = "/home/hquan07/Desktop/Real-Time_Stock_Prediction_Pipeline/src/data_ingestion/schema/stock_schema.avsc"
with open(SCHEMA_PATH, "r") as f:
    STOCK_SCHEMA = parse_schema(json.load(f))




def get_db_connection():
    """Create PostgreSQL connection"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
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
    logger.info(f"✅ Connected to PostgreSQL {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
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
