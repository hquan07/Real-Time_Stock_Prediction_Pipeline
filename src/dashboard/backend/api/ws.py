from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import json
import io
from fastavro import schemaless_reader, parse_schema
from aiokafka import AIOKafkaConsumer
from loguru import logger

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WS client: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

# Load Avro Schema
import os
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data_ingestion", "schema", "stock_schema.avsc")
with open(SCHEMA_PATH, "r") as f:
    SCHEMA = parse_schema(json.load(f))

async def consume_kafka():
    consumer = AIOKafkaConsumer(
        'stock_raw_data',
        bootstrap_servers='localhost:9092',
        group_id="fastapi_websocket_group"
    )
    # Get cluster layout and join group
    await consumer.start()
    logger.info("🟢 Kafka Consumer started for WebSocket broadcasting")
    try:
        async for msg in consumer:
            # Deserialize Avro
            bytes_reader = io.BytesIO(msg.value)
            try:
                record = schemaless_reader(bytes_reader, SCHEMA)
                
                # Format to match what frontend expects
                # The frontend's DashboardTab charts are for get_stock_data API
                # But here we stream real-time prices.
                payload = {
                    "type": "tick",
                    "ticker": record.get("ticker"),
                    "timestamp": record.get("timestamp"),
                    "price": record.get("price", {}).get("close")
                }
                
                await manager.broadcast(payload)
            except Exception as e:
                logger.error(f"Error deserializing or broadcasting message: {e}")
                
    except asyncio.CancelledError:
        pass
    finally:
        await consumer.stop()
        logger.info("🔴 Kafka Consumer stopped")

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
