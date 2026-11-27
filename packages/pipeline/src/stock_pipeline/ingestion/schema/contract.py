"""Semantic contract for stock-market events."""

from datetime import datetime, timezone
from hashlib import sha256
import re

import exchange_calendars as xcals
import pandas as pd


SCHEMA_VERSION = 2
DEFAULT_SOURCE = "yfinance"
DEFAULT_INTERVAL = "1d"
DEFAULT_CALENDAR = "XNYS"
DEFAULT_EXCHANGE_TIMEZONE = "America/New_York"
_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,19}$")


def normalize_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if not _TICKER_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid ticker: {ticker!r}")
    return normalized


def utc_millis(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("Event timestamps must be timezone-aware")
    return int(value.astimezone(timezone.utc).timestamp() * 1000)


def is_market_session(timestamp_ms: int, calendar_name: str = DEFAULT_CALENDAR) -> bool:
    timestamp = pd.Timestamp(timestamp_ms, unit="ms", tz="UTC")
    calendar = xcals.get_calendar(calendar_name)
    session_date = timestamp.tz_convert(DEFAULT_EXCHANGE_TIMEZONE).date()
    try:
        calendar.date_to_session(pd.Timestamp(session_date), direction="none")
        return True
    except ValueError:
        return False


def build_event_id(
    ticker: str,
    timestamp_ms: int,
    source: str = DEFAULT_SOURCE,
    interval: str = DEFAULT_INTERVAL,
) -> str:
    canonical = f"{normalize_ticker(ticker)}|{timestamp_ms}|{source}|{interval}"
    return sha256(canonical.encode("utf-8")).hexdigest()


def validate_ohlcv(price: dict) -> None:
    required = ("open", "high", "low", "close", "volume")
    missing = [field for field in required if price.get(field) is None]
    if missing:
        raise ValueError(f"Missing OHLCV fields: {missing}")
    if min(float(price[field]) for field in ("open", "high", "low", "close")) <= 0:
        raise ValueError("OHLC prices must be positive")
    if float(price["high"]) < max(float(price["open"]), float(price["close"])):
        raise ValueError("High price is below open/close")
    if float(price["low"]) > min(float(price["open"]), float(price["close"])):
        raise ValueError("Low price is above open/close")
    if int(price["volume"]) < 0:
        raise ValueError("Volume cannot be negative")
