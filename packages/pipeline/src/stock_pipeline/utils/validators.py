"""
Data validation utilities for Stock Prediction Pipeline.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union
import re

from loguru import logger


# ---------------------------------------------------------------------------
# Ticker Validation
class TickerValidator:
    """Validator for stock ticker symbols."""

    VALID_PATTERN = re.compile(r"^[A-Z0-9.-]{1,10}$")

    @staticmethod
    def validate(ticker: str) -> bool:
        """
        Validate ticker symbol format.
        
        Valid tickers:
            - 1-10 characters
            - Uppercase letters, numbers, dots, hyphens
            - Examples: AAPL, MSFT, BRK-B, BRK.B
        """
        if not ticker or not isinstance(ticker, str):
            return False

        ticker = ticker.strip().upper()
        return bool(TickerValidator.VALID_PATTERN.match(ticker))

    @staticmethod
    def validate_list(tickers: List[str]) -> List[str]:
        """
        Validate list of tickers and return only valid ones.
        """
        return [t.upper() for t in tickers if TickerValidator.validate(t)]


# ---------------------------------------------------------------------------
# Price Data Validation
# ---------------------------------------------------------------------------
class PriceValidator:
    """Validator for price data."""

    @staticmethod
    def validate_ohlcv(data: Dict[str, Any]) -> bool:
        """
        Validate OHLCV (Open, High, Low, Close, Volume) data.
        
        Rules:
            - All prices must be non-negative
            - High >= Low
            - High >= Open, Close
            - Low <= Open, Close
            - Volume must be non-negative
        """
        try:
            open_p = float(data.get("open", 0))
            high = float(data.get("high", 0))
            low = float(data.get("low", 0))
            close = float(data.get("close", 0))
            volume = int(data.get("volume", 0))

            # All must be non-negative
            if any(p < 0 for p in [open_p, high, low, close]):
                return False

            if volume < 0:
                return False

            # Price relationship checks
            if high < low:
                return False

            if high < open_p or high < close:
                return False

            if low > open_p or low > close:
                return False

            return True

        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_price_range(
        price: float,
        ticker: str = None,
        max_price: float = 1000000,
        min_price: float = 0.0001,
    ) -> bool:
        """
        Validate that price is within reasonable range.
        """
        if price is None:
            return False

        try:
            price = float(price)
            return min_price <= price <= max_price
        except (ValueError, TypeError):
            return False


# ---------------------------------------------------------------------------
# Date Validation
# ---------------------------------------------------------------------------
class DateValidator:
    """Validator for dates and timestamps."""

    @staticmethod
    def validate_trading_date(dt: date) -> bool:
        """
        Check if date is a valid trading day (weekday).
        Note: Does not check for market holidays.
        """
        if not isinstance(dt, date):
            return False

        # Weekday check (0=Monday, 6=Sunday)
        return dt.weekday() < 5

    @staticmethod
    def validate_date_range(
        start_date: date,
        end_date: date,
        max_days: int = 365 * 10,
    ) -> bool:
        """
        Validate date range.
        
        Rules:
            - start_date <= end_date
            - Range not exceeding max_days
            - Dates not in the future (for historical data)
        """
        if not isinstance(start_date, date) or not isinstance(end_date, date):
            return False

        if start_date > end_date:
            return False

        if (end_date - start_date).days > max_days:
            return False

        return True

    @staticmethod
    def validate_timestamp(ts: Union[int, float], unit: str = "ms") -> bool:
        """
        Validate Unix timestamp.
        
        Args:
            ts: Timestamp value
            unit: 'ms' for milliseconds, 's' for seconds
        """
        try:
            ts = float(ts)

            if unit == "ms":
                ts = ts / 1000

            # Valid range: 1970 to 2100
            dt = datetime.fromtimestamp(ts)
            return 1970 <= dt.year <= 2100

        except (ValueError, TypeError, OSError):
            return False


# ---------------------------------------------------------------------------
# Kafka Message Validation
# ---------------------------------------------------------------------------
class KafkaMessageValidator:
    """Validator for Kafka stock messages."""

    REQUIRED_FIELDS = ["ticker", "timestamp"]
    OPTIONAL_FIELDS = ["price", "dividends", "splits", "earnings", "financials"]

    @classmethod
    def validate(cls, message: Dict[str, Any]) -> bool:
        """
        Validate Kafka message structure.
        
        Required:
            - ticker: Valid ticker symbol
            - timestamp: Valid Unix timestamp
        
        Optional:
            - price: Valid OHLCV data
            - dividends: Numeric value
            - splits: Numeric value
            - earnings: Dict with eps data
            - financials: Dict with financial data
        """
        if not isinstance(message, dict):
            logger.warning("Message is not a dictionary")
            return False

        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in message:
                logger.warning(f"Missing required field: {field}")
                return False

        # Validate ticker
        if not TickerValidator.validate(message.get("ticker", "")):
            logger.warning("Invalid ticker format")
            return False

        # Validate timestamp
        if not DateValidator.validate_timestamp(message.get("timestamp", 0)):
            logger.warning("Invalid timestamp")
            return False

        # Validate price if present
        price = message.get("price")
        if price and not PriceValidator.validate_ohlcv(price):
            logger.warning("Invalid OHLCV data")
            return False

        return True


# ---------------------------------------------------------------------------
# Model Input Validation
# ---------------------------------------------------------------------------
class ModelInputValidator:
    """Validator for ML/DL model inputs."""

    @staticmethod
    def validate_features(
        features: List[float],
        expected_length: int = None,
        allow_nan: bool = False,
    ) -> bool:
        """
        Validate feature vector for model input.
        """
        import math

        if not isinstance(features, (list, tuple)):
            return False

        if expected_length and len(features) != expected_length:
            return False

        for f in features:
            if not isinstance(f, (int, float)):
                return False

            if not allow_nan and (math.isnan(f) or math.isinf(f)):
                return False

        return True

    @staticmethod
    def validate_prediction_output(
        prediction: Any,
        min_value: float = None,
        max_value: float = None,
    ) -> bool:
        """
        Validate model prediction output.
        """
        try:
            pred = float(prediction)

            if min_value is not None and pred < min_value:
                return False

            if max_value is not None and pred > max_value:
                return False

            return True

        except (ValueError, TypeError):
            return False


# ---------------------------------------------------------------------------
# Convenience validation functions
# ---------------------------------------------------------------------------
def is_valid_ticker(ticker: str) -> bool:
    """Shortcut for ticker validation."""
    return TickerValidator.validate(ticker)


def is_valid_price(price: float) -> bool:
    """Shortcut for price validation."""
    return PriceValidator.validate_price_range(price)


def is_valid_kafka_message(message: Dict) -> bool:
    """Shortcut for Kafka message validation."""
    return KafkaMessageValidator.validate(message)