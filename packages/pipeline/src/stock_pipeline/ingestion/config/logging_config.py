"""
Logging configuration for data ingestion module.
"""

import logging
import sys


def configure_logger(name: str = "data_ingestion", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
