import websocket
import json
from loguru import logger
from datetime import datetime
from src.data_ingestion.kafka_producer.send_data import send_to_kafka
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# We will need a Finnhub API Key. 
# We fetch it from .env or default to empty string so user can fill it.
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "WMT"]

def on_message(ws, message):
    data = json.loads(message)
    if data['type'] == 'trade':
        for trade in data['data']:
            symbol = trade['s']
            price = trade['p']
            volume = trade['v']
            timestamp = trade['t'] # milliseconds
            
            # Format the tick data into the expected Avro schema for the pipeline
            record = {
                "ticker": symbol,
                "timestamp": timestamp,
                "price": {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": int(volume) if volume else 1
                },
                "dividends": None,
                "splits": None,
                "earnings": None,
                "financials": None
            }
            
            logger.info(f"📈 [WebSocket] Received tick: {symbol} @ ${price:.2f}")
            send_to_kafka(data=record)

def on_error(ws, error):
    logger.error(f"❌ WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.warning("🔴 WebSocket Connection Closed")

def on_open(ws):
    logger.info("🟢 WebSocket Connection Opened")
    for ticker in TICKERS:
        ws.send(json.dumps({'type':'subscribe', 'symbol': ticker}))
        logger.info(f"Subscribed to {ticker}")

def run_finnhub_websocket():
    if not FINNHUB_API_KEY:
        logger.error("Vui lòng cung cấp FINNHUB_API_KEY trong file .env để chạy WebSocket!")
        return
        
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}",
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.on_open = on_open
    ws.run_forever()

if __name__ == "__main__":
    run_finnhub_websocket()
