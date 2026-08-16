#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.io import read_manifest
from data.preparation import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CODE-II for configurable ECG lead reconstruction")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-lead", default="V2")
    parser.add_argument("--target-rate", type=float)
    parser.add_argument("--baseline-correction", choices=("none", "median"), default="none")
    parser.add_argument(
        "--normalization", choices=("none", "per_record", "training_set"), default="training_set"
    )
    parser.add_argument("--split-mode", choices=("auto", "official", "random"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--summary-json", type=Path, default=Path("reports/preparation_summary.json"))
    args = parser.parse_args()
    summary = prepare_dataset(
        read_manifest(args.manifest),
        args.output,
        raw_root=args.raw_root,
        target_lead=args.target_lead,
        target_rate=args.target_rate,
        baseline_correction=args.baseline_correction,
        normalization=args.normalization,
        split_mode=args.split_mode,
        split_seed=args.seed,
        split_ratios=(args.train_fraction, args.validation_fraction, args.test_fraction),
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

