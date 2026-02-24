"""
Centralized logging configuration using Loguru.
Provides structured logging with file rotation and colored console output.
"""

import sys
import os
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
LOG_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# Log directory
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configure Loguru
# ---------------------------------------------------------------------------
def setup_logger(
    app_name: str = "stock_pipeline",
    log_level: str = None,
    enable_file: bool = True,
    enable_console: bool = True,
) -> logger:
    """
    Configure and return the Loguru logger.
    
    Args:
        app_name: Name of the application (used in log filenames)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file: Enable file logging with rotation
        enable_console: Enable colored console logging
    
    Returns:
        Configured logger instance.
    """
    # Remove default handler
    logger.remove()

    level = log_level or LOG_LEVEL

    # Console handler with colors
    if enable_console:
        logger.add(
            sys.stderr,
            format=LOG_FORMAT,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # File handler with rotation
    if enable_file:
        # Main log file with daily rotation
        logger.add(
            LOG_DIR / f"{app_name}.log",
            format=LOG_FILE_FORMAT,
            level=level,
            rotation="00:00",  # Rotate at midnight
            retention="30 days",  # Keep 30 days of logs
            compression="gz",  # Compress rotated files
            serialize=False,
            backtrace=True,
            diagnose=True,
        )

        # Error-only log file
        logger.add(
            LOG_DIR / f"{app_name}_errors.log",
            format=LOG_FILE_FORMAT,
            level="ERROR",
            rotation="1 week",
            retention="90 days",
            compression="gz",
            serialize=False,
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"✅ Logger configured for {app_name} at level {level}")
    return logger


# ---------------------------------------------------------------------------
# Pre-configured loggers for different modules
# ---------------------------------------------------------------------------
def get_ingestion_logger():
    """Get logger for data ingestion module."""
    return setup_logger("data_ingestion")


def get_streaming_logger():
    """Get logger for streaming module."""
    return setup_logger("streaming")


def get_ml_logger():
    """Get logger for machine learning module."""
    return setup_logger("machine_learning")


def get_dl_logger():
    """Get logger for deep learning module."""
    return setup_logger("deep_learning")


def get_dashboard_logger():
    """Get logger for dashboard module."""
    return setup_logger("dashboard")


# ---------------------------------------------------------------------------
# Initialize default logger on import
# ---------------------------------------------------------------------------
# Call setup on import to configure default logger
setup_logger()
