from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from physics_esn.pipeline import run_sub001_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sub-001 Wilson-Cowan EEG proof of concept.")
    parser.add_argument("--config", type=Path, default=Path("config") / "defaults.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    summary = run_sub001_pipeline(args.config, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))
