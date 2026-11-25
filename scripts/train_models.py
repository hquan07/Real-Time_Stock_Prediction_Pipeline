#!/usr/bin/env python3
"""Deprecated adapter for the canonical, leakage-safe P0 training CLI.

Use ``scripts/run_train.py`` for new automation. This module remains only so
existing local commands use the same validation and promotion gates as the
canonical implementation.
"""

from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "packages/pipeline/src"
sys.path.insert(0, str(PACKAGE_ROOT))

from stock_pipeline.cli.train import load_training_data, main, run_train


__all__ = ["load_training_data", "run_train"]


if __name__ == "__main__":
    main()
