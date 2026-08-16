#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.validation import validate_prepared_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automated checks on a prepared CODE-II HDF5 file")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target-lead", default="V2")
    parser.add_argument("--expected-rate", type=float)
    parser.add_argument("--report-json", type=Path, default=Path("reports/validation.json"))
    args = parser.parse_args()
    result = validate_prepared_dataset(args.dataset, args.target_lead, args.expected_rate)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

