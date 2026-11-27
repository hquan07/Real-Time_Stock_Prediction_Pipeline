"""Compatibility adapter for the canonical training CLI."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/pipeline/src"))

from stock_pipeline.cli.train import load_training_data, main, run_train

__all__ = ["load_training_data", "run_train"]


if __name__ == "__main__":
    main()
