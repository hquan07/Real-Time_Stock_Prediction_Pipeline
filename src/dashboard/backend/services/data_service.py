import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df = df.copy()
    
    # Moving Averages
    df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['ma50'] = df['close'].rolling(window=50, min_periods=1).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    
    return df

async def fetch_stock_data(pool, ticker: str, period: str = "3mo") -> pd.DataFrame:
    try:
        period_days = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "1y": 365, "3mo": 90}
        days = period_days.get(period, 90)
        start_date = datetime.now() - timedelta(days=days)
        
        if not pool:
            return pd.DataFrame()
            
        query = """
            SELECT event_date as date, open, high, low, close, volume
            FROM stock_prices_stream
            WHERE ticker = $1 AND event_date >= $2
            ORDER BY event_time ASC
        """
        async with pool.acquire() as conn:
            records = await conn.fetch(query, ticker, start_date.date())
            
        if records:
            df = pd.DataFrame([dict(r) for r in records])
            df = df.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
            df = calculate_indicators(df)
            # Ensure proper JSON serialization
            df.fillna(0, inplace=True)
            return df
    except Exception as e:
        logger.error(f"PostgreSQL async fetch failed: {e}")
    
    return pd.DataFrame()
