"""
Database connection module for Real-Time Stock Prediction Pipeline.
Provides SQLAlchemy engine, session management, and connection pooling.
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from loguru import logger


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "stockdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Huyquan1607")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ---------------------------------------------------------------------------
# Engine with connection pooling
# ---------------------------------------------------------------------------
_engine = None


def get_engine():
    """
    Create and return SQLAlchemy engine with connection pooling.
    Uses singleton pattern to reuse engine across calls.
    """
    global _engine

    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=False,
        )
        logger.info(f"✅ Database engine created for {DB_HOST}:{DB_PORT}/{DB_NAME}")

    return _engine


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
_SessionLocal = None


def get_session_factory():
    """Get or create session factory."""
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )

    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Automatically handles commit/rollback and session cleanup.
    
    Usage:
        with get_session() as session:
            session.execute(query)
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()

    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Database session error: {e}")
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------
def init_database(sql_file_path: str = None) -> bool:
    """
    Initialize database tables from SQL file.
    
    Args:
        sql_file_path: Path to SQL file with table definitions.
                      Defaults to init_tables.sql in same directory.
    
    Returns:
        True if successful, False otherwise.
    """
    if sql_file_path is None:
        sql_file_path = os.path.join(
            os.path.dirname(__file__),
            "init_tables.sql"
        )

    try:
        with open(sql_file_path, "r") as f:
            sql_content = f.read()

        engine = get_engine()

        with engine.connect() as conn:
            # Split by semicolon and execute each statement
            statements = [s.strip() for s in sql_content.split(";") if s.strip()]

            for stmt in statements:
                if stmt:
                    conn.execute(text(stmt))

            conn.commit()

        logger.info("✅ Database tables initialized successfully")
        return True

    except FileNotFoundError:
        logger.error(f"❌ SQL file not found: {sql_file_path}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        return False


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def check_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful, False otherwise.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection OK")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
def dispose_engine():
    """Dispose engine and close all connections in pool."""
    global _engine, _SessionLocal

    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionLocal = None
        logger.info("✅ Database engine disposed")
