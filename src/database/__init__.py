"""Deprecated compatibility namespace; use stock_pipeline.db."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / "packages/pipeline/src/stock_pipeline/db")]
