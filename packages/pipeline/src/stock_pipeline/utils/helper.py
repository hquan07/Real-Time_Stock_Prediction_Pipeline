"""
Common helper functions for the Stock Prediction Pipeline.
"""

import os
import json
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Environment Loading
# ---------------------------------------------------------------------------
def load_env(env_file: str = ".env") -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file
    
    Returns:
        True if file was loaded, False otherwise.
    """
    env_path = Path(env_file)

    if env_path.exists():
        load_dotenv(env_path)
        return True
    else:
        # Try to find .env in parent directories
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            possible_env = parent / ".env"
            if possible_env.exists():
                load_dotenv(possible_env)
                return True

    return False


def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Get environment variable with optional default and validation.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: Raise exception if not found
    
    Returns:
        Environment variable value or default.
    
    Raises:
        ValueError: If required and not found.
    """
    value = os.getenv(key, default)

    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' not set")

    return value


# ---------------------------------------------------------------------------
# Date/Time Helpers
# ---------------------------------------------------------------------------
def parse_date(date_str: str) -> Optional[date]:
    """
    Parse date string in various formats.
    
    Supported formats:
        - YYYY-MM-DD
        - DD/MM/YYYY
        - MM-DD-YYYY
    """
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    return None


def get_trading_days(start_date: date, end_date: date) -> List[date]:
    """
    Get list of trading days (weekdays) between two dates.
    Note: Does not account for market holidays.
    """
    days = []
    current = start_date

    while current <= end_date:
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            days.append(current)
        current += timedelta(days=1)

    return days


def format_timestamp(ts: Union[int, float], unit: str = "ms") -> datetime:
    """
    Convert Unix timestamp to datetime.
    
    Args:
        ts: Unix timestamp
        unit: 'ms' for milliseconds, 's' for seconds
    
    Returns:
        Datetime object.
    """
    if unit == "ms":
        ts = ts / 1000

    return datetime.fromtimestamp(ts)


def datetime_to_timestamp(dt: datetime, unit: str = "ms") -> int:
    """
    Convert datetime to Unix timestamp.
    
    Args:
        dt: Datetime object
        unit: 'ms' for milliseconds, 's' for seconds
    
    Returns:
        Unix timestamp.
    """
    ts = dt.timestamp()

    if unit == "ms":
        return int(ts * 1000)

    return int(ts)


# ---------------------------------------------------------------------------
# Data Helpers
# ---------------------------------------------------------------------------
def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    """
    Flatten nested dictionary.
    
    Example:
        {"a": {"b": 1}} -> {"a_b": 1}
    """
    items = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split list into chunks of specified size.
    
    Example:
        chunk_list([1,2,3,4,5], 2) -> [[1,2], [3,4], [5]]
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ---------------------------------------------------------------------------
# File Helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(file_path: Union[str, Path]) -> Optional[Dict]:
    """Load JSON file and return contents."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_json(data: Dict, file_path: Union[str, Path], indent: int = 2) -> bool:
    """Save data to JSON file."""
    try:
        ensure_dir(Path(file_path).parent)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stock-specific Helpers
# ---------------------------------------------------------------------------
def normalize_ticker(ticker: str) -> str:
    """
    Normalize stock ticker symbol.
    
    - Convert to uppercase
    - Remove whitespace
    - Handle common variations
    """
    if not ticker:
        return ""

    ticker = ticker.strip().upper()

    # Handle common variations
    ticker = ticker.replace(".", "-")  # BRK.B -> BRK-B

    return ticker


def is_valid_ticker(ticker: str) -> bool:
    """
    Basic validation for stock ticker.
    
    Returns True if ticker looks valid (1-6 uppercase letters/numbers).
    """
    import re

    if not ticker:
        return False

    ticker = normalize_ticker(ticker)
    # Valid tickers: 1-6 characters, letters, numbers, and hyphens
    pattern = r"^[A-Z0-9-]{1,6}$"

    return bool(re.match(pattern, ticker))


def calculate_return(price_old: float, price_new: float) -> Optional[float]:
    """Calculate percentage return."""
    if price_old is None or price_new is None or price_old == 0:
        return None

    return ((price_new - price_old) / price_old) * 100
