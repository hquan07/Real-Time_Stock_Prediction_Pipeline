"""Compatibility adapter for the canonical research-signal CLI."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/pipeline/src"))

from stock_pipeline.cli.signals import compute_rsi, run_signal_generation

__all__ = ["compute_rsi", "run_signal_generation"]


if __name__ == "__main__":
    run_signal_generation()
