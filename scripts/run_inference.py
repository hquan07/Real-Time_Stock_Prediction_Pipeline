"""Compatibility adapter for the canonical inference CLI."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/pipeline/src"))

from stock_pipeline.cli.inference import load_latest_data, main, run_inference

__all__ = ["load_latest_data", "run_inference"]


if __name__ == "__main__":
    main()
