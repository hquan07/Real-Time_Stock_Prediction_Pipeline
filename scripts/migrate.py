#!/usr/bin/env python3
"""Thin CLI adapter for versioned database migrations."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/pipeline/src"))

from stock_pipeline.db.migrations import apply_migrations, rollback_latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("up", "down"), default="up", nargs="?")
    args = parser.parse_args()
    if args.action == "up":
        for migration in apply_migrations():
            print(f"applied {migration}")
    else:
        print(f"rolled back {rollback_latest()}")


if __name__ == "__main__":
    main()
