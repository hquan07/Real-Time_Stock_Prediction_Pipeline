"""Deprecated compatibility namespace; use stock_pipeline.integrations."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "packages/pipeline/src/stock_pipeline/integrations")]
